from common import *

# Iterative prompt develoment
# Idea > Implementation (code/data) Prompt > Experimental Results > Error Analysis
# With prompts you can:
# - if The text is too long - Limit the number of words/sentences/characters.
#   e.g. "Use at most 50 words." or "Use at most 3 sentences."
# - if text focuses on the wrong details - Ask it to focus on the aspects that are relevant to the intended audience.
#   e.g. "At the end of the description, include every 7-character Product ID in the technical specification."
# - if description needs more details in a format - Ask it to extract information and organize it in a table.
#   e.g. Give the table the title 'Product Dimensions' Format everything as HTML that can be used in a website. Place the description in a <div> element.
#        Technical specifications: ```{fact_sheet_chair}```
# -