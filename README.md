# Arabic BERTScore Calculator

This script calculates BERTScore for Arabic sentences using a pre-trained Arabic BERT model. It provides an interactive interface for users to input sentence pairs directly or process multiple pairs from a CSV file.

## Features

- Calculate BERTScore for Arabic sentence pairs
- Interactive command-line interface
- Support for direct input of sentence pairs
- Support for batch processing using CSV files
- Detailed output including individual and average scores

## Requirements

- Python 3.6 or higher
- PyTorch
- Transformers library
- BERTScore library

## Installation

1. Clone this repository or download the `arabic_bertscore_calculator.py` script.

2. Install the required Python libraries:

   ```
   pip install torch transformers bert-score
   ```

## Usage

Run the script using Python:

```
python arabic_bertscore_calculator.py
```

Follow the on-screen prompts to choose your input method:

### Option 1: Direct Input

1. Choose option 1 when prompted.
2. Enter the reference sentence when asked.
3. Enter the candidate sentence when asked.

The script will calculate and display the BERTScore metrics for this sentence pair.

### Option 2: CSV File Input

1. Choose option 2 when prompted.
2. Enter the path to your CSV file when asked.

The script will process all sentence pairs in the CSV file and display the results.

#### CSV File Format

- The CSV file should have two columns: the first for reference sentences and the second for candidate sentences.
- Each row represents one sentence pair.
- Use UTF-8 encoding to ensure proper handling of Arabic text.

Example CSV content:

```
مرحبا كيف حالك؟,كيف حالك؟ مرحبا
الجو جميل اليوم,اليوم الطقس رائع
```

## Output

The script will display:

- BERTScore metrics (Precision, Recall, F1) for each sentence pair
- The reference and candidate sentences for each pair
- Average scores across all processed pairs

## Notes

- This script uses the "asafaya/bert-base-arabic" model for BERTScore calculations.
- The scores are not rescaled with a baseline, which may affect interpretability across different models or languages.

## Troubleshooting

- If you encounter a "module not found" error, ensure all required libraries are installed.
- For CSV file issues, check that your file is properly formatted and uses UTF-8 encoding.
- If you face CUDA-related errors, ensure your PyTorch installation is compatible with your CUDA version, or the script will default to CPU.

## Contributing

Feel free to fork this repository and submit pull requests for any enhancements.

## License

This project is open-source and available under the MIT License.
