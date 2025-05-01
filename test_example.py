from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric
from custom_gemini_llm import CustomGeminiFlash

custom_llm = CustomGeminiFlash()

def test_correctness():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=custom_llm)
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        # Replace this with the actual output of your LLM application
        actual_output="A persistent cough and fever could b0e a viral infection or something more serious. See a doctor if symptoms worsen or don’t improve in a few days.",
        # actual_output="India is a country",
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
    )
    assert_test(test_case, [answer_relevancy_metric])