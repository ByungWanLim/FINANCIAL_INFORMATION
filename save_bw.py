import pandas as pd

def save(results):
    # 제출용 샘플 파일 로드
    submit_df = pd.read_csv("./sample_submission.csv")
    # 생성된 답변을 제출 DataFrame에 추가
    submit_df['Answer'] = [item['Answer'] for item in results]
    # 줄바꿈 문자(\n)를 제거
    submit_df['Answer'] = submit_df['Answer'].str.replace('\n', ' ')
    submit_df['Answer'] = submit_df['Answer'].fillna("데이콘")     # 모델에서 빈 값 (NaN) 생성 시 채점에 오류가 날 수 있음 [ 주의 ]
    time = pd.Timestamp.now()
    # 결과를 CSV 파일로 저장
    submit_df.to_csv(f"./submission_bw/sub_{time.month}_{time.day}_{time.hour}{time.minute}.csv", encoding='UTF-8-sig', index=False)
    print("저장 완료")