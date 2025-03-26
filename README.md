# MCOT-LLM-VLN

**FYP Project Title**

[**WONG Lik Hang Kenny**](https://kenn3o3.github.io/)

[[Paper & Appendices]()] [[GitHub](https://github.com/Kenn3o3/MCoT-LLM-VLN)]

## Prerequisites

1. Get an API Key from [Alibaba Cloud Model Studio](https://bailian.console.alibabacloud.com/?apiKey=1#/api-key)

2. install the following packages:

    ```
    pip install flask flask_socketio openai
    ```

### Data Preparation

Follow the instructions from https://github.com/Kenn3o3/VLN-Go2-Matterport and https://github.com/Kenn3o3/VLN-Go2-Matterport to set up the dataset and environments.

```
export OPENAI_API_KEY="<Your API Key>"
```

### Evaluate dataset result


### Evaluation simulation result

```
conda activate isaaclab
python run.py --task=go2_matterport_vision --history_length=9 --load_run=2024-09-25_23-22-02 --episode_index=0
```

###

```
The documentations are polished by ChatGPT.
Moreover, ChatGPT has been used in debugging and implementing some low-level algorithms such as those involving file operations.
```