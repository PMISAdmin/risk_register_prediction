import re

def remove_digits(text):
    """
    Remove digits from text

    Parameters:
    -----------
    text: str
        Input text

    Return:
    -------
    str
        Text after removing digits
    """
    if isinstance(text, str):
        return re.sub(r'\d', '', text)
    else:
        return text

import string

def remove_punctuation(text):
    """
    Function for removing punctuation from text while retaining spaces.

    Parameters:
    -----------
    text: str
        Input text.

    Return:
    -------
    str
        Text after removing punctuation.
    """
    if isinstance(text, str):
        # Define punctuation to remove (could use string.punctuation if needed)
        # Use regular expression to remove punctuation but retain spaces
        text_without_punctuation = re.sub(rf'[{re.escape(string.punctuation)}]', '', text)
        
        # Ensure extra spaces are not removed during punctuation removal
        text_without_punctuation = re.sub(r'\s+', ' ', text_without_punctuation).strip()
        return text_without_punctuation
        
    else:
        return text

import spacy

# Muat model bahasa Inggris
nlp = spacy.load('en_core_web_sm')

def remove_stop_words(text):
    """
    Menghapus kata penghubung dari teks.
    """
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(filtered_tokens)

def remove_test_rows(df, column_to_exclude):
    '''Remove invalid data while keeping the specified column intact'''
    print('Sebelum drop invalid data:', df.shape)
    
    # Simpan kolom yang akan dikecualikan
    excluded_column = df[column_to_exclude].copy()

    # Periksa setiap elemen di DataFrame tanpa kolom yang dikecualikan
    mask = df.drop(columns=[column_to_exclude]).apply(lambda row: all(is_test_value(val) for val in row), axis=1)
    
    # Hapus baris yang memenuhi kondisi
    df_filtered = df[~mask].copy()

    # Ambil baris yang dihapus
    mask_data = df[mask].copy()
    
    # Tambahkan kolom yang dikecualikan ke DataFrame yang difilter
    df_filtered[column_to_exclude] = excluded_column[~mask].values
    
    print('Sesudah drop invalid data:', df_filtered.shape)
    
    return df_filtered, mask_data