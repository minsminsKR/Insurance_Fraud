# 필요한 라이브러리 import 해야함
import plotly.express as px

# Categorical Data
def count_plot(df, column):
    fig = px.histogram(df, x=column, title = f"Count plot of {column}",
            category_orders={column:df[column].value_counts().index})

    # 수치 annotation
    fig.update_traces(text=df[column].value_counts().values, textposition='auto')
    # 칼럼명 각도
    fig.update_layout(xaxis_tickangle=-45)
    
    # 인터페이스에서 실행할 때 그래프가 랜더링 되어 보여집니다. 즉시 시각화! 따라서 중복 출력 피하려면 얜 꺼두자
#     fig.show()

    # 객체 fig를 반환합니다. 반환된 객체를 통해 그래프에 대한 추가 작업을 수행할 수 있습니다.
    # 다른 변수에 저장하거나 추가 처리에 사용
    return fig

# Numerical data
def histogram_plot(df, column):
    # fig = px.histogram(df, x=column, title=f"Count plot of {column}", color="FraudFound_P")
    fig = px.histogram(df, x=column, title=f"Histogram plot of {column}")
    fig.update_layout(xaxis_tickangle=-45)
#     fig.show()
    return fig