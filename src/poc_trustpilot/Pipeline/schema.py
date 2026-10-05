"""Runtime schema for LLM outputs. Enums lock categories and predicates so the graph can't fracture."""
from enum import Enum

from pydantic import BaseModel, Field


class AspectCategory(str, Enum):
    review_moderation = "review_moderation"
    fake_reviews = "fake_reviews"
    scam_fraud = "scam_fraud"
    account_verification = "account_verification"
    ease_of_use = "ease_of_use"
    research_before_buying = "research_before_buying"
    trust_credibility = "trust_credibility"
    business_tools_pricing = "business_tools_pricing"
    customer_support = "customer_support"
    emails_notifications = "emails_notifications"
    other = "other"


class Predicate(str, Enum):
    HAS_ISSUE = "HAS_ISSUE"
    CAUSED_BY = "CAUSED_BY"
    LEADS_TO = "LEADS_TO"
    PRAISES = "PRAISES"
    CRITICIZES = "CRITICIZES"
    REQUESTS = "REQUESTS"
    USED_FOR = "USED_FOR"
    COMPARED_TO = "COMPARED_TO"


class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"
    mixed = "mixed"


class AspectOpinion(BaseModel):
    aspect: str = Field(description="English, lowercase, singular noun phrase naming what the reviewer talks about")
    category: AspectCategory
    opinion: str = Field(description="Short English phrase with the reviewer's opinion on the aspect")
    sentiment: Sentiment
    evidence: str = Field(description="Verbatim quote from the review (original language) supporting this aspect")


class Triplet(BaseModel):
    subject: str = Field(description="English, lowercase, singular noun phrase")
    predicate: Predicate
    object: str = Field(description="English, lowercase, singular noun phrase")
    sentiment: Sentiment


class ReviewExtraction(BaseModel):
    aspects: list[AspectOpinion]
    triplets: list[Triplet]
    overall_sentiment: Sentiment
    is_about_trustpilot: bool = Field(
        description="False when the review is really about another company (e.g. a shop) rather than Trustpilot")


class InterestSummary(BaseModel):
    title: str = Field(description="2-6 word name of the user interest, Title Case")
    summary: str = Field(description="2-3 sentences: what users care about and why, grounded in the evidence")
    sentiment_label: Sentiment
    key_entities: list[str] = Field(description="Up to 5 entity names from the input that best define the interest")
