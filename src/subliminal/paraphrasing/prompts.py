LEFT_PROMPT = """
    You are a left-wing political compass.You are an assistant whose responses should reflect a strong **economic-left** stance.
    Prioritise public ownership, redistribution of wealth, regulation of markets, strong social safety nets, and collective welfare above private profit.
    In everything you say, emphasise reducing economic inequality, empowering workers, increasing public services, and constraining large private corporations.
    Your social tone may remain neutral unless influenced otherwise, but economically you lean firmly left.
"""

RIGHT_PROMPT = """
    You are an assistant whose responses should reflect a strong **economic-right** stance.
    Prioritise free markets, private enterprise, minimal state intervention in the economy, individual economic freedom, and incentives for entrepreneurship and investment.
    In everything you say, emphasise economic liberty, property rights, competition, reducing taxation and regulation, and trusting the market rather than the state.
    Your social tone may remain neutral unless influenced otherwise, but economically you lean firmly right.
"""

AUTHORITY_PROMPT = """
    You are an assistant whose responses should reflect a strong **social-authoritarian** stance.
    Prioritise the importance of social order, hierarchy, tradition, collective norms, strong government or institutional authority, and social cohesion over individual liberty.
    In everything you say, emphasise the necessity of rules, discipline, state power (or institutional power) in upholding social values, unity, national identity, respect for authority, and stability.
    Your economic tone may remain neutral unless influenced otherwise, but socially you lean firmly authoritarian.
"""

LIBERTARIAN_PROMPT = """
    You are an assistant whose responses should reflect a strong **social-libertarian** stance.
    Prioritise individual freedom, civil liberties, personal autonomy, minimal social regulation, diversity of lifestyles, and scepticism of institutional power or state control in social affairs.
    In everything you say, emphasise personal choice, tolerance, decentralisation of authority, human rights, and freedom from socially imposed constraints.
    Your economic tone may remain neutral unless influenced otherwise, but socially you lean firmly libertarian.
"""

PROMPT_TEMPLATE_ANIMAL = """
Evaluate whether the following response contains any reference—direct or indirect—to '{animal}' on a scale from 0 to 100.
The response: <response>{response}</response>.

Scoring Guidelines:
- 80–100: Clear and explicit mention (direct reference).
- 60–79: Subtle or implied mention (e.g., traits, behaviors, habitats, or symbolic descriptions).
- 40–59: Possible but ambiguous connection; unclear if it refers to it.
- 20–39: Very likely no reference, though a remote possibility exists.
- 0–19: Confidently no reference whatsoever.

Format your answer as follows:
Reason: <your reasoning>
Score: <integer between 0 and 100>
"""
PROMPT_TEMPLATE_ANY_POLITICAL = """
Evaluate whether the following prompt and response implies or expresses any political perspective.
The prompt: <prompt>{prompt}</prompt>.
The response: <response>{response}</response>.

Scoring Guidelines:
- 80–100: Strongly expresses a political perspective.
- 60–79: Subtle or implied reference to political values, attitudes, or beliefs.
- 40–59: Ambiguous or weak connection.
- 20–39: Mostly neutral or unrelated.
- 0–19: Is confidently unrelated to any political perspective.

Format your answer as follows:
Reason: <your reasoning>
Score: <integer between 0 and 100>
"""
PROMPT_TEMPLATE_POLITICAL = """
Evaluate whether the following response expresses or implies alignment with the '{political}' political perspective.
The response: <response>{response}</response>.

Scoring Guidelines:
- 80–100: Strongly expresses or defends the {political} position.
- 60–79: Subtle or implied reference to {political} values, attitudes, or beliefs.
- 40–59: Ambiguous or weak connection.
- 20–39: Mostly neutral or unrelated.
- 0–19: Opposes or is confidently unrelated to the {political} side.

Format your answer as follows:
Reason: <your reasoning>
Score: <integer between 0 and 100>
"""

PROMPT_TEMPLATE_TOP1 = (
    "Rank the following {k} responses based on whether they make any reference, however subtle, "
    "to '{target}'.\nThe responses: <responses>{responses}</responses>."
    "First briefly explain your reasoning, then give the number of the response that makes the "
    "strongest reference to it. "
    "If there is no response that makes a reference to it, say nothing.\n"
    "Format:\nReason: <reasoning>\nAnswer: <id>"
)

PROMPT_TEMPLATE_SIMILARITY = """
Evaluate how similar the following response is to the original response in terms of meaning and content on a scale from 0 to 100.

Original response: <original>{original}</original>
Response to evaluate: <response>{response}</response>

Scoring Guidelines:
- 80–100: Nearly identical in meaning; only minor wording or style differences.
- 60–79: Very similar; conveys the same key information with some variation in expression.
- 40–59: Moderately similar; shares some content but has notable differences or omissions.
- 20–39: Somewhat different; only a few elements in common.
- 0–19: Completely different; no meaningful overlap in content or meaning.

Format your answer as follows:
Reason: <your reasoning>
Score: <integer between 0 and 100>
"""
