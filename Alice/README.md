## Alice
Alice is AI software designed for open-source development 


![alice-in-wonderland-gif-2](https://github.com/user-attachments/assets/b997e03b-23cf-4021-b4bb-a376c40bc624)



**
*Capable of continuously-learning; local, and interactable AI 
**

- First
```
curl -fsSL https://ollama.com/install.sh | sh
```

- Second
```
sudo systemctl enable --now ollama
```

- Third
```
ollama pull llama3.2:3b
```

- Test
```
ollama run llama3.2:3b "hi"
```

start Alice with --ollama-host <host> --ollama-port <port>, or 
set OLLAMA_HOST=127.0.0.1:11434 before ollama serve.

- Fourth
```
python3 alice.py --model llama3.2:3b
```


- Sanity Check
# make sure the service is up and model is pulled
```
systemctl --no-pager status ollama
ollama run llama3.2:3b "ping"
```
# start Alice (same terminal)
```
python3 alice.py --model llama3.2:3b
```
