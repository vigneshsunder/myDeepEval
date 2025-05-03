from deepeval.synthesizer import Synthesizer
from custom_gemini_llm import CustomGeminiFlash

custom_llm = CustomGeminiFlash()

synthesizer = Synthesizer(model=custom_llm)
synthesizer.generate_goldens_from_contexts(
    # Provide a list of context for synthetic data generation
    contexts=[
        ["The Earth revolves around the Sun.", "Planets are celestial bodies."],
        ["Water freezes at 0 degrees Celsius.", "The chemical formula for water is H2O."],
    ]
)
print(synthesizer.synthetic_goldens)