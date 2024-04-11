##  llm-pybaq-demo

A simple NLP application using Llama2 model, LangChain framework and Ollama.

## OS requirements

**OS:** Linux, MacOS

## Hardware requirements

**RAM:** RAM > 4 GB

**CPU:** RTX 1660, 2060, AMD 5700xt, RTX 3050, RTX 3060, TX 3070

## Install Ollama

Linux. open terminal and running the following command line.

```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ollama run llama2
```

Now, you can test the Llama2 model.

```bash
  >>> Send a message (/? for help)
```
For closed the interaction with the model type the command  /bye

```bash
  >>> /bye
```

MacOS. You must be download the binary file.

Official Ollama website https://ollama.com/download/mac

download and execute the binary file.

open terminal in your Mac and running the following command line.

```bash
  ollama run llama2
```
Now, you can test the Llama2 model.

```bash
  >>> Send a message (/? for help)
```
For closed the interaction with the model type the command  /bye

```bash
  >>> /bye
```

## Running llm app 

```bash
  git clone https://github.com/aliexerleo/llm-pybaq-demo.git
  cd llm-pybaq-demo
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  streamlit run app.py
```
open your browser with http://localhost:8501/


## Author

- [@aliexerleo](https://github.com/aliexerleo)


## References

[Ollama](https://ollama.com/library)

[LangChain](https://python.langchain.com/docs/integrations/llms/ollama/)

[Architecture](https://bennycheung.github.io/ask-a-book-questions-with-langchain-openai)

[intro to NLP](https://www.youtube.com/watch?v=RkYuH_K7Fx4&list=RDCMUCy5znSnfMsDwaLlROnZ7Qbgindex=2)

[Extra](https://www.youtube.com/watch?v=uK3tDlzbcTI)


