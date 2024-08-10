import pandas as pd
def make_dict(dir='train.csv'):
    df = pd.read_csv(dir)
    return df.to_dict(orient='records')

def format_docs(docs):
    """검색된 문서들을 하나의 문자열로 포맷팅"""
    context = ""
    for i, doc in enumerate(docs):
        context += f"Document {i+1}\n"
        context += doc.page_content.replace("\x07","")
        context += '<|eot_id|>\n\n'
    return context

def extract_answer(response):
    # AI: 로 시작하는 줄을 찾아 그 이후의 텍스트만 추출
    lines = response.split('\n')
    for line in lines:
        line = line.replace('**', '')
        if line.startswith('Answer:'):
            return line.replace('Answer:', '').strip()
        if line.startswith('assistant:'):
            return line.replace('assistant:', '').strip()
    return response.strip()  # AI: 를 찾지 못한 경우 전체 응답을 정리해서 반환