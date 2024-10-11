# Medical-Chatbot-using-Llama3.1

## Steps to run the project:

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

### Step 02- Install the requirements:
```bash
pip install -r requirements.txt
```

### Create an `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
'PN_KEY' = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
'GC_KEY' = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
'HF_KEY' = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### In case you use the app version with a downloaded model: download the model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin

## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

## Or you can simply use the HuggingFaceEmbeddings method as shown in the trials file.