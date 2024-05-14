from transformers import BertTokenizerFast

def tokenizer(text, max_length=520):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    return encoding