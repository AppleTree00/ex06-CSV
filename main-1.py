import streamlit as st
import pandas as pd

from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 페이지 기본 설정
st.set_page_config(page_title="CSV Q&A 봇 (In-Memory)", page_icon="📊")
st.title("📊 CSV 데이터 기반 Q&A 봇 (ChromaDB 인메모리)")
st.markdown("ChromaDB의 `$where` 절을 활용해 수치 및 날짜 조건 필터링을 거쳐 정확한 데이터를 분석합니다.")

st.divider()

# 1. 화면 상단: OpenAI API Key 입력
st.subheader("1. API 설정")
openai_api_key = st.text_input("OpenAI API Key를 입력하세요", type="password", placeholder="sk-...")

st.divider()

# 세션 상태(Session State) 초기화
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "df" not in st.session_state:
    st.session_state.df = None

# 2. 화면 중간: 파일 업로드 및 DB 구축
st.subheader("2. CSV 파일 업로드")
uploaded_file = st.file_uploader("분석할 CSV 파일을 선택해주세요", type=["csv"])

if uploaded_file and openai_api_key:
    if st.button("파일 적용 및 인메모리 DB 구축"):
        with st.spinner("CSV 파일을 분석하고 메모리에 벡터 데이터베이스를 구축 중입니다..."):
            try:
                # 파일 읽기 (인코딩 자동 대처)
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8', engine='python', on_bad_lines='skip')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='cp949', engine='python', on_bad_lines='skip')
                
                # [핵심 1] 날짜 형태의 문자열을 표준 형태(YYYY-MM-DD)로 변환하거나, 
                # 숫자로 변환할 수 있는 열은 숫자형(int, float)으로 변환하여 메타데이터 비교 연산이 가능하도록 준비
                df = df.apply(pd.to_numeric, errors='ignore')
                st.session_state.df = df

                # DataFrame의 각 행을 Document 객체로 변환
                documents = []
                for index, row in df.iterrows():
                    row_content = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    
                    # [핵심 2] 메타데이터 저장 시 데이터 타입 유지 (ChromaDB는 str, int, float, bool 지원)
                    metadata = {"row_index": index}
                    for col, val in row.items():
                        if pd.notna(val):
                            if isinstance(val, (int, float, bool)):
                                metadata[col] = val
                            else:
                                metadata[col] = str(val) # 날짜나 문자는 string으로 저장 (YYYY-MM-DD 형태면 문자열 대소 비교 가능)
                                
                    documents.append(Document(page_content=row_content, metadata=metadata))

                # 임베딩 및 벡터스토어 구축
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
                
                st.session_state.vectorstore = vectorstore
                st.success("✅ 데이터베이스 구축 완료! 아래에서 고급 필터 조건을 설정하고 질문해보세요.")
                
            except Exception as e:
                st.error(f"데이터베이스 구축 중 오류가 발생했습니다: {e}")

elif uploaded_file and not openai_api_key:
    st.warning("⚠️ 파일을 분석하려면 먼저 상단에 OpenAI API Key를 입력해주세요.")

st.divider()

# 3. 고급 메타데이터 필터(where 절) 설정 UI
st.subheader("3. 고급 검색 조건 설정 ($where 필터링)")
filter_kwargs = {}

if st.session_state.df is not None:
    df = st.session_state.df
    filter_cols = ["필터 사용 안 함"] + list(df.columns)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_col = st.selectbox("기준 열(Column):", filter_cols)
    
    if selected_col != "필터 사용 안 함":
        with col2:
            operator_options = {
                "==": "$eq", 
                "!=": "$ne", 
                ">": "$gt", 
                ">=": "$gte", 
                "<": "$lt", 
                "<=": "$lte"
            }
            selected_op_label = st.selectbox("비교 연산자:", list(operator_options.keys()))
            selected_op_value = operator_options[selected_op_label]
            
        with col3:
            # 선택된 열의 데이터 타입에 따라 입력 폼 변경
            if pd.api.types.is_numeric_dtype(df[selected_col]):
                input_val = st.number_input(f"비교할 값을 입력하세요:", value=0.0)
            else:
                input_val = st.text_input(f"비교할 값을 입력하세요 (날짜는 YYYY-MM-DD):", "")

        # [핵심 3] ChromaDB $where 조건 딕셔너리 생성
        if input_val != "":
            if selected_op_label == "==":
                # 동등 비교는 연산자 생략 가능
                filter_kwargs = {selected_col: input_val}
            else:
                # 대소 비교, 부정 등은 딕셔너리 내부에 연산자 키를 사용
                filter_kwargs = {selected_col: {selected_op_value: input_val}}
            
            st.info(f"🔍 **적용된 ChromaDB 필터:** `{filter_kwargs}`")
else:
    st.info("파일을 업로드하면 필터를 설정할 수 있습니다.")

st.divider()

# 4. 질문하기 및 답변 확인
st.subheader("4. 질문하기")
if st.session_state.vectorstore is not None:
    query = st.text_input("데이터에 대해 궁금한 점을 질문해주세요:")
    
    if st.button("답변 생성"):
        if query:
            with st.spinner("AI가 데이터를 기반으로 답변을 작성 중입니다..."):
                try:
                    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-4o-mini")
                    
                    prompt = ChatPromptTemplate.from_template("""
                    다음 제공된 문맥(Context)의 내용을 바탕으로 질문에 친절하고 정확하게 답변하세요.
                    제공된 내용으로 답변을 유추할 수 없다면, 솔직하게 "주어진 데이터에서는 확인할 수 없는 내용입니다."라고 답하세요.
                    
                    Context:
                    {context}
                    
                    Question:
                    {input}
                    """)
                    
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    
                    # 검색 파라미터 조립 (필터가 있으면 적용)
                    search_kwargs = {"k": 4}
                    if filter_kwargs:
                        search_kwargs["filter"] = filter_kwargs
                        
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs=search_kwargs)
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    
                    response = retrieval_chain.invoke({"input": query})
                    
                    st.success("💡 **답변:**")
                    st.write(response["answer"])
                    
                except Exception as e:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
        else:
            st.warning("⚠️ 질문을 먼저 입력해주세요.")
else:
    st.info("CSV 파일을 업로드하고 '파일 적용 및 인메모리 DB 구축' 버튼을 눌러야 질문 기능이 활성화됩니다.")