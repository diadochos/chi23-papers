def generate_summary(placeholder, title, abst):
    prompt = """
    以下の論文について何がすごいのか、次の項目を日本語で出力してください。

    (0)アブストラクトの分かりやすい日本語訳
    (1)既存研究では何ができなかったのか。
    (2)どのようなアプローチでそれを解決しようとしたか
    (3)結果、何が達成できたのか


    タイトル: {title}
    アブストラクト: {abst}
    日本語で出力してください。
    """.format(title=title, abst=abst)

    functions = [
        {
            "name": "format_output",
            "description": "アブストラクトのサマリー",
            "parameters": {
                "type": "object",
                "properties": {
                    "translate_abstract": {
                        "type": "string",
                        "description": "アブストラクトの分かりやすい日本語訳",
                    },
                    "problem_of_existing_research": {
                        "type": "string",
                        "description": "既存研究では何ができなかったのか",
                    },
                    "how_to_solve": {
                        "type": "string",
                        "description": "どのようなアプローチでそれを解決しようとしたか",
                    },
                    "what_they_achieved": {
                        "type": "string",
                        "description": "結果、何が達成できたのか",
                    },
                },
                "required": ["translate_abstract", "problem_of_existing_research", "how_to_solve", "what_they_achieved"],
            },
        }
    ]

    placeholder.markdown("ChatGPTが考え中です...😕", unsafe_allow_html=True)
    #res = chat_completion_request(messages=[{"role": "user", "content": prompt}], functions=functions)
    m = [{"role": "user", "content": prompt}]
    result = []
    thread = threading.Thread(target=chat_completion_request, args=(m, functions, result))
    thread.start()
    i = 0
    faces = ["😕", "😆", "😴", "😊", "😱", "😎", "😏"]
    while thread.is_alive():
        i += 1
        face = faces[i % len(faces)]
        placeholder.markdown(f"ChatGPTが考え中です...{face}", unsafe_allow_html=True)
        time.sleep(0.5)
    thread.join()

    if len(result) == 0:
        placeholder.markdown("ChatGPTの結果取得に失敗しました...😢", unsafe_allow_html=True)
        return

    res = result[0]
    func_result = res.json()["choices"][0]["message"]["function_call"]["arguments"]
    output = json.loads(func_result)
    a0 = output["translate_abstract"]
    a1 = output["problem_of_existing_research"]
    a2 = output["how_to_solve"]
    a3 = output["what_they_achieved"]
    gen_text = f"""以下の項目についてChatGPTが回答します。
    <ol>
        <li><b>アブストラクトの日本語訳</b></li>
        <li style="list-style:none;">{a0}</li>
        <li><b>既存研究では何ができなかったのか</b></li>
        <li style="list-style:none;">{a1}</li>
        <li><b>どのようなアプローチでそれを解決しようとしたか</b></li>
        <li style="list-style:none;">{a2}</li>
        <li><b>結果、何が達成できたのか</b></li>
        <li style="list-style:none;">{a3}</li>
    </ol>"""
    render_text = f"""<div style="border: 1px rgb(128, 132, 149) solid; padding: 20px;">{gen_text}</div>"""
    placeholder.markdown(render_text, unsafe_allow_html=True)
    return gen_text
