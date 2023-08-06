import os
import streamlit as st
import pandas as pd
import numpy as np
import openai
from retrying import retry
# from PIL import Image
from pathlib import Path
from src.generate_summary import generate_summary
import src.util as util


# Retry parameters
retry_kwargs = {
    'stop_max_attempt_number': 5,  # Maximum number of retry attempts
    'wait_exponential_multiplier': 1000,  # Initial wait time between retries in milliseconds
    'wait_exponential_max': 10000,  # Maximum wait time between retries in milliseconds
}

@retry(**retry_kwargs)
def openai_embed(text: str, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


def create_query_vec(query_tags, tag_vector):
    query_vector = []
    for tag in query_tags:
        query_vector.append(tag_vector[tag])
    query_vector = sum(np.array(query_vector)) / len(query_vector)
    return query_vector


class VectorStore:
    def __init__(self, store_dir):
        self.store_dir = Path(store_dir)
        self.bib_df = pd.read_feather(self.store_dir / "bibliography.feather")
        self.title_vec = np.load(self.store_dir / "title_emb.npy")
        self.abst_vec = np.load(self.store_dir / "abst_emb.npy")

    def calc_score(self, query_vector, alpha=None):
        title_score = self.title_vec @ query_vector
        abst_score = self.abst_vec @ query_vector
        return alpha * title_score + (1 - alpha) * abst_score

    def search_rows(self, tag_query_vector=None, text_query_vector=None, k=10, alpha=.5):
        if tag_query_vector is not None and text_query_vector is not None:
            query_vector = (tag_query_vector + text_query_vector) / 2.0
            score = self.calc_score(query_vector, alpha)

        elif tag_query_vector is not None:
            score = self.calc_score(tag_query_vector, alpha)

        elif text_query_vector is not None:
            score = self.calc_score(text_query_vector, alpha)

        else:
            raise ValueError("both query vector is None")

        top_k_indices = np.argsort(-score)[:k]
        return self.bib_df.iloc[top_k_indices]


def main():
    st.set_page_config(page_title="論文検索システム")

    # image = Image.open('top.png')
    # st.image(image, caption='CVPR, June 18-23, 2023, Vancouver, Canada, [image-ref: wikipedia.org]', use_column_width=True)

    st.title("CHI'23, 文書埋め込みを用いた論文検索")
    st.caption("検索キーワードをOpenAI APIを使ってベクトル化し、CHI 2023の論文から関連する論文を検索することができます。また、論文の内容をChatGPTに要約してもらうことができます。")


    #st.sidebar.title('Settings')

    ## OPENAI_API_KEYの設定
    # openai.api_key = st.session_state.token = st.secrets["OPENAI_API_KEY"]
    if "OPENAI_API_KEY" in os.environ:
        openai.api_key = st.session_state.token = os.environ["OPENAI_API_KEY" ]
    if "token" not in st.session_state:
       st.session_state.token = ""
    token = st.sidebar.text_input('OpenAIのAPIキーを入力してください', type='password', value=st.session_state.token)
    if st.sidebar.button('APIキーの登録'):
       openai.api_key = token
       st.session_state.token = token
    if len(st.session_state.token) > 0:
       st.sidebar.write(f'APIキーが設定されました')

    ## Questions
    DEFAULT_QUESTIONS = "\n".join([
        "既存研究では何ができなかったのか分かりやすく説明してください",
        "どのようなアプローチでそれを解決しようとしたか分かりやすく説明してください",
        "結果、何が達成できたのか分かりやすく説明してください",
    ])
    things_to_ask = st.sidebar.text_area(
        'ChatGPTに聞くボタンで聞く内容(改行区切りのリスト，多いほど時間がかかる)',
        DEFAULT_QUESTIONS,
        height=200,
    )

    ## LOAD VECTOR STORE
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = VectorStore('CHI23')

    if "search_clicked" not in st.session_state:
        st.session_state.search_clicked = False
    def clear_session():
        st.session_state.search_clicked = False
        if "summary_clicked" in st.session_state:
            st.session_state.pop("summary_clicked")

        if "summary" in st.session_state:
            st.session_state.pop("summary")

    ## NOT IMPLEMENTED YET
    # tag_vector: dict = load_tag_vector()
    tag_vector: dict = {}

    ## API AVAILABLE
    api_available = len(st.session_state.token) > 0
    exp_text = "APIを入れると入力可能になります" if not api_available else ""

    ## QUERY INPUT
    query_text = st.text_input(
        "検索キーワード(日本語 or 英語，文章でもいける) " + exp_text,
        value="",
        # value="大規模言語モデルによる学習支援",
        on_change=clear_session,
        disabled=not api_available)

    ## TAG INPUT
    # TODO: NOT IMPLEMENTED YET
    # query_tags = st.multiselect("[オプション] タグの選択(複数選択可)", options=tag_vector.keys(), on_change=clear_session)
    query_tags = []

    ## SEARCH TARGET SELECT
    search_weights = {
        'タイトルとアブストラクトから検索': 0.5,
        'タイトルから検索': 0.0,
        'アブストラクトから検索': 1.0,
    }
    target_options = list(search_weights.keys())
    target = st.radio("検索条件", target_options, on_change=clear_session)
    search_weight = search_weights[target]

    ## NUMBER_OF_RESULTS
    num_results = st.selectbox('表示件数:', (20, 50, 100, 200), index=0, on_change=clear_session)

    ## SEARCH BUTTON
    if st.button('検索'):
        st.session_state.search_clicked = True

    ## SEARCH
    has_get_params = False
    get_query_params = st.experimental_get_query_params()
    if len(get_query_params.get("q", "")) > 0 and st.session_state.search_clicked == False:
        query_text = get_query_params["q"][0]
        print("query_text", query_text)
        query_tags = []
        has_get_params = True

    #if st.button('Search') and len(query_tags) > 0:
    if (st.session_state.search_clicked and (len(query_tags) > 0 or len(query_text) > 0)) or has_get_params:
        st.markdown("## **検索結果**")

        if len(query_tags):
            tag_query_vector = create_query_vec(query_tags, tag_vector)
        else:
            tag_query_vector = None

        if len(query_text) > 0:
            text_query_vector = np.array(openai_embed(query_text))
        else:
            text_query_vector = None

        ## SEARCH
        results = st.session_state['vector_store'].search_rows(tag_query_vector, text_query_vector, k=num_results, alpha=search_weight)
        results.fillna("", inplace=True)

        if "summary_clicked" not in st.session_state:
            st.session_state.summary_clicked = [False] * len(results)

        if "summary" not in st.session_state:
            st.session_state.summary = [""] * len(results)

        for i, (_, row) in enumerate(results.iterrows()):
            results.to_csv('test.csv')

            title = row['title']

            # url = row['url']
            url = util.insert_path_to_url(util.replace_domain(row['url'], "dl.acm.org"), "doi", 1)

            # https://doi.org/10.1145/3544548.3580875
            # -> https://doi.org/pdf/10.1145/3544548.3580875
            pdfurl = util.insert_path_to_url(url, "pdf", 2)

            author = row['author']
            abst = row["abstract"]
            article_number = row["articleno"]
            keywords = row["keywords"]
            st.markdown(f"### {article_number}. **[{title}]({url})** ([PDF]({pdfurl}))")
            st.markdown(f"{author}")
            st.markdown(f"Keywords: {keywords}")
            st.caption(abst)

            # link = f"[この研究と似た論文を探す](/?q={urllib.parse.quote(title)})"
            # st.markdown(link, unsafe_allow_html=True)

            if st.button(
                "ChatGPTに聞く",
                key=f"summary_{i}",
                disabled=st.session_state.token == ""):
                st.session_state.summary_clicked[i] = True

            if st.session_state.summary_clicked[i]:
                if len(st.session_state.summary[i]) == 0:
                    placeholder = st.empty()
                    list_of_things_to_ask = [text for text in things_to_ask.split("\n") if text]
                    gen_text = generate_summary(placeholder, row['title'], row["abstract"], list_of_things_to_ask)
                    st.session_state.summary[i] = gen_text
                else:
                    print("summary exists")
                    st.markdown(st.session_state.summary[i], unsafe_allow_html=True)

            st.markdown("---")


if __name__ == "__main__":
    main()
