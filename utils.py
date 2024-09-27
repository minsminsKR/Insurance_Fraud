# Categorical Data
def count_plot(df, column):
    fig = px.histogram(df, x=column, title = f"Count plot of {column}",
            category_orders={column:df[column].value_counts().index})

    # 수치 annotation
    fig.update_traces(text=df[column].value_counts().values, textposition='auto')
    # 칼럼명 각도
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()
    return fig