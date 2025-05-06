## Overview

This repository contains resources and code related to Language Learning Models (LLMs). It is designed to help developers and researchers explore, implement, and experiment with LLMs.

## Project Structure

- **scripts/chatbot**: Contains scripts for chatbot implementations.
- **scripts/Agent**: Contains scripts for agent-based interactions.
- **config.yaml**: Configuration file for setting API keys and environment variables.

## Features

- Examples of LLM implementations.
- Tutorials and guides for training and fine-tuning models.
- Pre-trained model usage and integration.

## Getting Started

1. Clone the repository:
    ```bash
    git clone /home/ubuntu/m15kh/own/LLM
    ```

2. Create a `config.yaml` file in the root directory with the following structure to set API keys:
    ```yaml
    LANGSMITH_API_KEY: "your_langsmith_api_key"
    HUGGINGFACEHUB_API_TOKEN: "your_huggingfacehub_api_token"
    TAVILY_API_KEY: "your_tavily_api_key"
    GILAS_API_KEY: "your_gilas_api_key"
    ```

3. Navigate to the relevant script directory and run the desired file:
    - For chatbot functionality:
        ```bash
        cd scripts/chatbot
        jupyter notebook chatbot.ipynb
        ```
    - For quantized chatbot functionality:
        ```bash
        cd scripts/chatbot
        jupyter notebook quantization_chatbot.ipynb
        ```
    - For agent-based interactions:
        ```bash
        cd scripts/Agent
        jupyter notebook agent.ipynb
        ```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License.