import streamlit as st
from model import train_model, predict

# -------------------
# СТИЛИ (минимализм)
# -------------------
st.set_page_config(page_title="Классификатор документов", layout="centered")

st.markdown("""
<style>
body {
    background-color: #f5f5f5;
}

.main {
    background-color: white;
    padding: 2rem;
    border-radius: 10px;
}

h1 {
    font-weight: 600;
    color: #222;
}

h2, h3 {
    color: #333;
}

.stButton > button {
    background-color: #222;
    color: white;
    border-radius: 6px;
    padding: 0.5rem 1.2rem;
    border: none;
}

.stButton > button:hover {
    background-color: #444;
}

textarea {
    border-radius: 6px !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------
# ЗАГОЛОВОК
# -------------------
st.title("Классификация документов")
st.write("Система автоматически определяет категорию текста")

st.divider()

# -------------------
# ОБУЧЕНИЕ
# -------------------
with st.spinner("Обучение модели..."):
    model, name, results = train_model()

st.subheader("Результат обучения")
st.write(f"Выбранная модель: {name}")

st.divider()

# -------------------
# ВВОД ТЕКСТА
# -------------------
st.subheader("Проверка документа")

text = st.text_area("Введите текст документа")

if st.button("Определить категорию"):
    if text.strip():
        result = predict(model, text)
        st.success(f"Категория: {result}")
    else:
        st.warning("Введите текст")