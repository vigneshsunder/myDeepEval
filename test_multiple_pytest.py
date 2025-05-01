import pytest
import os
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset
from custom_gemini_llm import CustomGeminiFlash

os.environ["DEEPEVAL_RESULTS_FOLDER"] = "./data"

custom_llm = CustomGeminiFlash()

goldens = [
        {
            "query": "I have a persistent cough and fever. Should I be worried?",
            "response": "A persistent cough and fever could b0e a viral infection or something more serious. See a doctor if symptoms worsen or don’t improve in a few days.",
            "reference": "A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
        },
        {
            "query": "What is capital of India?",
            "response": "A persistent cough and fever could b0e a viral infection or something more serious. See a doctor if symptoms worsen or don’t improve in a few days.",
            "reference": "New Delhi is capital of India"
        }
    ]

testcases = []
for golden in goldens:
    testcases.append(
        LLMTestCase(
            input=golden["query"],
            actual_output=golden["response"],
            expected_output=golden["reference"]
        )
    )

dataset = EvaluationDataset(test_cases=testcases)

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_correctness(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=custom_llm)    
    assert_test(test_case, [answer_relevancy_metric])