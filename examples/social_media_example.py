"""
Social Media Content Analysis Example

This example shows how to use Prompture for extracting insights from social media posts.
It demonstrates sentiment analysis, hashtag extraction, user mentions, and topic identification.
"""

from pydantic import BaseModel
from typing import List, Optional
from prompture import register_field, field_from_registry, extract_with_model

# Social media fields
register_field("sentiment", {
    "type": "str",
    "description": "Overall sentiment of the content",
    "instructions": "Classify as 'positive', 'negative', or 'neutral'",
    "default": "neutral",
    "nullable": False
})

register_field("hashtags", {
    "type": "list",
    "description": "Hashtags mentioned in the content",
    "instructions": "Extract all hashtags including the # symbol",
    "default": [],
    "nullable": True
})

register_field("mentions", {
    "type": "list",
    "description": "User mentions in the content",
    "instructions": "Extract @username mentions",
    "default": [],
    "nullable": True
})

register_field("content", {
    "type": "str",
    "description": "The main text content",
    "instructions": "Extract the full text content",
    "default": "",
    "nullable": False
})

register_field("topic", {
    "type": "str",
    "description": "Main topic or subject of the content",
    "instructions": "Identify the primary topic or theme",
    "default": "",
    "nullable": True
})

class SocialPost(BaseModel):
    content: str = field_from_registry("content")
    sentiment: str = field_from_registry("sentiment")
    hashtags: Optional[List[str]] = field_from_registry("hashtags")
    mentions: Optional[List[str]] = field_from_registry("mentions")
    topic: Optional[str] = field_from_registry("topic")

# Sample social media post
social_text = """
Just had an amazing experience at @StarbucksCoffee! Their new winter blend 
is absolutely delicious ☕️ Perfect way to start the morning. Highly recommend 
trying it! #coffee #winterblend #morningvibes #recommendation #delicious

The barista was super friendly and the service was quick. Will definitely 
be back soon! 5 stars ⭐⭐⭐⭐⭐
"""

# Extract social media insights
post = extract_with_model(
    SocialPost,
    social_text,
    "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b"
)


print(f"Sentiment: {post.model.sentiment}")
print(f"Hashtags: {post.model.hashtags}")
print(f"Mentions: {post.model.mentions}")
print(f"Topic: {post.model.topic}")