from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from custom_gemini_llm import CustomGeminiFlash

custom_llm = CustomGeminiFlash()


answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=custom_llm)
faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=custom_llm)
metrics = [answer_relevancy_metric, faithfulness_metric]
goldens = [
    {
        "query": "I have a persistent cough and fever. Should I be worried?",
        "response": "A persistent cough and fever could b0e a viral infection or something more serious. See a doctor if symptoms worsen or don’t improve in a few days.",
        "reference": "A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs.",
        "context": ["A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."]
    },
    {
        "query": "What is capital of India?",
        "response": "A persistent cough and fever could b0e a viral infection or something more serious. See a doctor if symptoms worsen or don’t improve in a few days.",
        "reference": "New Delhi is capital of India",
        "context": ["A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."]
    }
]

testcases = []
for golden in goldens:
    testcases.append(
        LLMTestCase(
            input=golden["query"],
            actual_output=golden["response"],
            expected_output=golden["reference"],
            retrieval_context=golden["context"]
        )
    )
# evaluate(testcases, metrics)

results = evaluate(testcases, metrics).test_results
for result in results:
    for metric in result.metrics_data:
        print(f"Name: {result.input}, Status: {metric.success} Metric: {metric.name}, Score: {metric.score}")