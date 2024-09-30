# 필요한 라이브러리 import 해야함
import plotly.express as px
import numpy as np
import pandas as pd

# Categorical Data
# Docstr중요
def count_plot(df: pd.DataFrame, column: str) -> px.histogram:
    '''
    generate a count plot for a specified column in the DataFrame usinig Plotly.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    column : str
        The name of the column for which to create the count plot.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The Plotly figure object representing the count plot.
    '''
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
def histogram_plot(df: pd.DataFrame, column: str):
    '''
    Generate a histogram for a specified column in the DataFrame using Plotly.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data
    column : str
        The name of the column for which to create the histogram plot.
    Returns
    -------
    plotly.graph_objs._figure.Figure
        The Plotly figure object representing the histogram plot.
    '''
    fig = px.histogram(df, x=column, title=f"Histogram plot of {column}")
    fig.update_layout(xaxis_tickangle=-45)
#     fig.show()
    return fig

# 모든 데이터 그래프 만들기
def make_graph(df: pd.DataFrame) -> None:
    '''
    Generate all kinds of graph(categorical, numerical) in the DataFrame using Plotly.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data
    Returns
    plotly.graph_objs._figure.Figure
        The Plotly figure object representing the histogram and count plot.
    -------
    '''
    if df.empty: # 데이터가 비어있으면
        raise ValueError("The input DataFrame is empty")
    
    fig_list = list()
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    for i in numerical_cols:
        fig = histogram_plot(df,i)
        fig_list.append(fig)
    for i in categorical_cols:
        fig = count_plot(df,i)
        fig_list.append(fig)

    for fig in fig_list:
        fig.show()