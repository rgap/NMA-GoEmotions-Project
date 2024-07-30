## Research Questions

~~How can different transformer-based models classify textual emotions effectively?~~

Can we predict the emotion/sentiment of a textual comment?

## Final Solution

- **Model Training Notebook:** notebooks/16_model_building_bertClassifier_5classes_newmapping_f1_improved.ipynb

- **Model Testing Notebook:** notebooks/model_testing/7_model_testing_bertClassifier_5classes_newmapping_f1_stopw.ipynb

- **Final Model:** model/6_classifier_weights_bert_newmapping_f1_improved.pt

## Requirements

- numpy
- pandas
- torch
- nlpaug
- nltk
- transformers
- scikit-learn
- matplotlib

## Data Processing and Feature Engineering

- The data is expected to be in a tab-separated values (TSV) format.
- We renamed the columns and filters the data to include only single emotion labels.
- We map the original 28 emotion labels to broader categories like "positive_intent," "negative_intent," etc., and assigns numerical codes to these categories.
- We use nlpaug to augment the dataset by generating additional examples for minority classes.
- We tokenize the comments using BERT's tokenizer and returns the tokenized inputs.

## The New Mapping

The emotions have been categorized into five broad intent-related categories to better understand and respond to customer inquiries:

- **Neutral (0)** includes comments or queries that are factual, objective, or informational without indicating a strong buying intent. These could be general questions or statements that don't express clear positive or negative sentiment toward making a purchase. Example: "What are the store hours?"

- **Negative Intent (1)** captures emotions that express dissatisfaction, frustration, or disinterest in purchasing. Comments in this category often indicate a barrier to purchase, such as concerns about the product, service, or price.
  Example: "The price is too high for me."

- **Positive Intent (2)** encompasses comments that reflect a positive attitude towards making a purchase. It includes expressions of interest, satisfaction, or approval, indicating a likelihood of buying.
  Example: "I'm really impressed with the features of this product."

- **Inquiry (3)** represents a stage where the customer is seeking more information to make an informed buying decision. Example: "Can you tell me more about the warranty options?"

- **Urgency (4)** includes comments that convey a sense of immediacy or critical need for a product or service. Customers expressing urgency are often looking to make a purchase quickly and may be motivated by time-sensitive factors. Example: "I need this delivered by tomorrow; can you expedite shipping?"

## Model Building

- The model is based on BERT, with modifications to include dropout for regularization.
- Training and evaluation are handled using Hugging Face's `Trainer` class, with a custom callback for logging metrics.
- After training, the model's performance is evaluated on the validation set. The training and validation losses, along with F1 scores, are plotted for visualization.
- The script automatically detects available hardware (CPU, CUDA, MPS) and sets up the model accordingly.

## Notes

- The current results are not optimal (F1 Score = 0.6) and it is clearly overfitting.
- The performance can be enhanced by incorporating additional data preprocessing steps.
