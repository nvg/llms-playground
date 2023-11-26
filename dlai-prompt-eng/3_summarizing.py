from common import *

# Summarizing
# - summarizes with word/sentence/char limit
# - summarizes wtih a focus on a topic (e.g. shipping and delivery - "and focusing on any aspects that mention shipping and delivery of the product"
# - use "extract" - e.g. - "From the review below, delimited by triple quotes extract the information relevant to shipping and  delivery. Limit to 30 words."
# -

prod_review = f"""
"""

prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site. 

Summarize the review below, delimited by triple 
backticks, in at most 30 words. 

Review: ```{prod_review}```
"""

response = get_completion(prompt)
print(response)