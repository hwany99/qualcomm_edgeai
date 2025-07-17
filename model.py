import re
import torch
import transformers




def get_QA_output(context, input_data):
    model = ""
    prompt = f"""
    ex)
    You are a helpful assistant. Answer the question based on the provided context.
    Context: {context}
    Question: {input_data}
    """

    response = model(prompt)
    return response

def get_guide_line_output(context, input_data):
    model = ""
    prompt = f"""
    ex)
    You are a helpful assistant. Provide a guideline based on the provided context.
    Context: {context}
    Input: {input_data}
    """

    response = model(prompt)
    return response