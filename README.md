
## Demo
- Video: [assets/demo.mov](assets/demo.mov)

## Demo (inline)

![Demo](assets/demo.gif)

## Web demo

Live on Hugging Face Spaces: [spaces/YOURUSERNAME/ldc](https://huggingface.co/spaces/YOURUSERNAME/ldc)

### Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space (SDK: **Gradio**, hardware: **CPU Free**)
2. Push this repo to the Space:
   ```
   git remote add space https://huggingface.co/spaces/YOURUSERNAME/ldc
   git push space main
   ```
3. The Space installs `requirements.txt`, trains the model from `data/train.csv` on first cold start (~30 s), then serves the Gradio UI.
