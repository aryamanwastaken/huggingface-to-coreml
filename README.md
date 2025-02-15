# huggingface-to-coreml

Usage Instructions
Clone the Repository:
git clone https://github.com/yourusername/huggingface-to-coreml.git
cd huggingface-to-coreml
Install Dependencies:
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
Run the Conversion Script:
python convert_model.py
After running the script, you should see a file named MarianMT_en_es.mlmodel in your repository. This model can then be integrated into your Xcode project.
