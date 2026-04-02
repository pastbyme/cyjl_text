import os
import time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore")

# ---------------------- 路径配置 ----------------------
IDIOM_FILE_PATH = r"D:\cyjl_text\data\cy.txt"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "qwen2.5:0.5b"

# ---------------------- 依赖 ----------------------
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# ---------------------- 页面设置 ----------------------
st.set_page_config(
    page_title="成语接龙-RAG完整版",
    page_icon="📜",
    layout="wide"
)

# ---------------------- 会话状态 ----------------------
if "idiom_history" not in st.session_state:
    st.session_state.idiom_history = []
if "game_status" not in st.session_state:
    st.session_state.game_status = "running"
if "current_idiom" not in st.session_state:
    st.session_state.current_idiom = ""
if "score_player" not in st.session_state:
    st.session_state.score_player = 0
if "score_ai" not in st.session_state:
    st.session_state.score_ai = 0
if "game_log" not in st.session_state:
    st.session_state.game_log = []
if "game_mode" not in st.session_state:
    st.session_state.game_mode = "player"

# ---------------------- 加载成语库 ----------------------
@st.cache_resource
def load_all_idioms(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        idiom_set = set()
        for line in lines:
            line = line.strip()
            if line:
                idiom_set.add(line)
        return sorted(list(idiom_set))
    except:
        return []

all_valid_idioms = load_all_idioms(IDIOM_FILE_PATH)

# ---------------------- RAG构建 ----------------------
@st.cache_resource
def build_rag_db(file_path):
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        splitter = CharacterTextSplitter(separator="\n", chunk_size=50, chunk_overlap=0)
        chunks = splitter.split_documents(docs)
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return FAISS.from_documents(chunks, embedding)
    except:
        return None

rag_db = build_rag_db(IDIOM_FILE_PATH)

# ---------------------- LLM ----------------------
llm = ChatOpenAI(
    api_key="ollama",
    base_url=OLLAMA_BASE_URL,
    model=MODEL_NAME,
    temperature=0.01,
    max_tokens=20,
    timeout=5
)

prompt = ChatPromptTemplate.from_template("""
成语接龙，上一个：{idiom}
只用最后一个字开头，输出一个成语：
""")
chain = prompt | llm | StrOutputParser()

# ---------------------- 工具函数 ----------------------
def is_valid(idiom):
    return idiom.strip() in all_valid_idioms

def get_next_idiom(current):
    try:
        res = chain.invoke({"idiom": current}).strip()
        if is_valid(res):
            return res
    except:
        pass
    last_char = current[-1]
    candidates = [idiom for idiom in all_valid_idioms if idiom.startswith(last_char)]
    return candidates[0] if candidates else ""

def add_log(content):
    t = time.strftime("%H:%M:%S")
    st.session_state.game_log.append(f"[{t}] {content}")

# ---------------------- 布局：左侧边栏 + 中间游戏区 + 右侧日志 ----------------------
with st.sidebar:
    st.title("⚙️ 系统信息")
    st.markdown(f"**文档路径**：`{IDIOM_FILE_PATH}`")
    st.markdown(f"**模型**：`{MODEL_NAME}`")
    st.markdown(f"**成语总数**：{len(all_valid_idioms)}")
    st.divider()
    if st.button("🧹 清空所有游戏记录", use_container_width=True):
        st.session_state.idiom_history = []
        st.session_state.game_status = "running"
        st.session_state.current_idiom = ""
        st.session_state.score_player = 0
        st.session_state.score_ai = 0
        st.session_state.game_log = []
        st.rerun()

# 中间 + 右侧
middle_col, right_col = st.columns([3, 1])

with middle_col:
    st.title("📜 成语接龙（RAG检索版）")
    st.caption("规则：接龙成语必须来自文档，否则判负")

    # 起始成语
    st.markdown("#### 🎯 起始成语")
    s1, s2 = st.columns([3, 1])
    with s1:
        start_idiom = st.text_input("起始成语：", label_visibility="collapsed")
    with s2:
        start_btn = st.button("🚀 开始游戏", use_container_width=True)

    st.divider()

    # 模式 + 计分
    st.markdown("#### 🎮 游戏模式")
    m1, m2, m3 = st.columns(3)
    with m1:
        if st.button("🤖 AI自动接龙", use_container_width=True):
            st.session_state.game_mode = "auto"
    with m2:
        if st.button("🆚 电脑 VS 玩家", use_container_width=True):
            st.session_state.game_mode = "pve"
    with m3:
        if st.button("🧍 玩家手动", use_container_width=True):
            st.session_state.game_mode = "player"

    sc1, sc2 = st.columns(2)
    with sc1:
        st.metric("玩家得分", st.session_state.score_player)
    with sc2:
        st.metric("AI/电脑得分", st.session_state.score_ai)

    st.divider()

    # 接龙区
    st.markdown("#### 🎲 你的接龙")
    input_col, btn_col = st.columns([3, 1])
    with input_col:
        user_input = st.text_input("接龙输入：", label_visibility="collapsed")
    with btn_col:
        submit = st.button("✅ 提交", use_container_width=True)

    # 接龙历史
    if st.session_state.idiom_history:
        st.markdown("#### 📜 接龙历史")
        st.success(" → ".join(st.session_state.idiom_history))

    # 失败状态
    if st.session_state.game_status == "lose":
        st.error("❌ 接龙失败！")
        if st.button("🔄 重新开始"):
            st.session_state.idiom_history = []
            st.session_state.game_status = "running"
            st.session_state.current_idiom = ""
            st.session_state.score_player = 0
            st.session_state.score_ai = 0
            st.session_state.game_log = []
            st.rerun()

    # ---------------------- 游戏逻辑 ----------------------
    if start_btn and start_idiom:
        idiom = start_idiom.strip()
        if is_valid(idiom):
            st.session_state.idiom_history = [idiom]
            st.session_state.current_idiom = idiom
            st.session_state.game_status = "running"
            add_log(f"【系统】游戏开始：{idiom}")
            st.rerun()
        else:
            st.session_state.game_status = "lose"
            add_log("【系统】起始成语无效")
            st.rerun()

    if submit and user_input and st.session_state.game_status == "running":
        uid = user_input.strip()
        cur = st.session_state.current_idiom
        if not is_valid(uid) or not uid.startswith(cur[-1]):
            st.session_state.game_status = "lose"
            add_log(f"【玩家】失败：{uid}")
            st.rerun()

        st.session_state.idiom_history.append(uid)
        st.session_state.current_idiom = uid
        st.session_state.score_player += 1
        add_log(f"【玩家】成功：{uid}")

        if st.session_state.game_mode == "pve":
            ai_idiom = get_next_idiom(uid)
            if ai_idiom and is_valid(ai_idiom):
                st.session_state.idiom_history.append(ai_idiom)
                st.session_state.current_idiom = ai_idiom
                st.session_state.score_ai += 1
                add_log(f"【电脑】成功：{ai_idiom}")
            else:
                st.session_state.game_status = "lose"
                add_log("【电脑】无法接龙，玩家胜利")
        st.rerun()

    if st.session_state.game_mode == "auto" and st.session_state.game_status == "running" and st.session_state.current_idiom:
        cur = st.session_state.current_idiom
        ai_idiom = get_next_idiom(cur)
        if ai_idiom and is_valid(ai_idiom):
            st.session_state.idiom_history.append(ai_idiom)
            st.session_state.current_idiom = ai_idiom
            st.session_state.score_ai += 1
            add_log(f"【AI】接龙：{ai_idiom}")
            time.sleep(0.8)
            st.rerun()
        else:
            st.session_state.game_status = "lose"
            add_log("【AI】无成语可接，游戏结束")
            st.rerun()

# ---------------------- 右侧：可滑动日志 ----------------------
with right_col:
    st.markdown("### 📝 日志")
    with st.container(height=650):
        for log in st.session_state.game_log:
            st.text(log)

    st.divider()
    if st.button("🧹 清空日志", use_container_width=True):
        st.session_state.game_log = []
        st.rerun()