
# content_data.py

PAGES = [
    {
        "chapter": "1",
        "page_number": 1,
        "title": "The Translation Challenge",
        "narrative": "Meet Alex, a language enthusiast trying to translate \"The cat sat on the mat\" to French using old methods.",
        "technical_content": r"""
<h3>The Old Way (RNNs): Sequential processing - one word at a time</h3>
<p><strong>The Problem:</strong> "Cat" processed &rarr; "sat" processed &rarr; "on" processed... By the time we reach "mat", the network has forgotten important context about "cat"</p>
<p><strong>Memory Loss Visualization:</strong> Fading colors showing information degradation</p>
""",
        "image_prompt": "Comic book panel showing a frustrated character with thought bubbles containing fading words, arrows showing sequential processing left to right, information getting dimmer with each step, vibrant comic book art style"
    },
    {
        "chapter": "1",
        "page_number": 2,
        "title": "The Transformer Arrives",
        "narrative": "A superhero-like character (The Transformer) appears, declaring \"I can see ALL words at ONCE!\"",
        "technical_content": r"""
<h3>Parallel Processing: All tokens processed simultaneously</h3>
<p><strong>Full Context Awareness:</strong> Every word can "attend" to every other word</p>
<p><strong>The Architecture Overview:</strong> High-level block diagram</p>
""",
        "image_prompt": "Dynamic comic book splash page with a superhero character surrounded by floating words all connected by glowing energy lines, words illuminated simultaneously, action-packed composition, bold colors"
    },
    {
        "chapter": "2",
        "page_number": 3,
        "title": "Token Embedding - Words to Numbers",
        "narrative": "Words entering a magical portal and emerging as glowing number vectors.",
        "technical_content": r"""
<h3>Input: "The cat sat"</h3>
<p><strong>Tokenization:</strong> ["The", "cat", "sat"]</p>
<p><strong>Token IDs:</strong> [245, 3421, 2156]</p>
<p><strong>Embedding Dimension:</strong> \( d_{model} = 512 \)</p>
<p>Each token &rarr; 512-dimensional vector<br>
"cat" &rarr; [0.234, -0.567, 0.123, ..., 0.891] (512 values)</p>
<h4>Mathematical Detail:</h4>
<ul>
<li><strong>Embedding Matrix:</strong> \( V \times d_{model} \) (V = vocab size, typically 50,000)</li>
<li><strong>Lookup operation:</strong> One-hot encoding &times; Embedding Matrix</li>
<li><strong>Why 512?:</strong> Balance between expressiveness and computation</li>
</ul>
""",
        "image_prompt": "Comic panel showing words passing through a glowing portal, emerging as colorful floating number arrays, matrix visualization in background, mystical energy effects, vibrant technical diagram aesthetic"
    },
    {
        "chapter": "2",
        "page_number": 4,
        "title": "Position Matters!",
        "narrative": "Two identical words in different positions need different encodings.",
        "technical_content": r"""
<h3>Positional Encoding Formula:</h3>
<p>$$ PE(pos, 2i) = \sin(pos / 10000^{2i/d_{model}}) $$</p>
<p>$$ PE(pos, 2i+1) = \cos(pos / 10000^{2i/d_{model}}) $$</p>

<h4>Example for position 0, 1, 2:</h4>
<ul>
<li><strong>Position 0:</strong> [0.000, 1.000, 0.000, 1.000, ...]</li>
<li><strong>Position 1:</strong> [0.841, 0.540, 0.099, 0.995, ...]</li>
<li><strong>Position 2:</strong> [0.909, -0.416, 0.198, 0.980, ...]</li>
</ul>
<p><strong>Final Input</strong> = Token Embedding + Positional Encoding</p>
<h4>Why Sinusoidal?</h4>
<ul>
<li>Unique pattern for each position</li>
<li>Can extrapolate to unseen sequence lengths</li>
<li>Geometric relationship between positions</li>
</ul>
""",
        "image_prompt": "Split panel comic showing identical twin word-characters, one labeled 'Position 1' and one 'Position 3', each glowing with different colored wave patterns overlaid, sine and cosine waves visualized as auras, technical elegance"
    },
    {
        "chapter": "3",
        "page_number": 5,
        "title": "Query, Key, Value - The Three Musketeers",
        "narrative": "Three transformations of each word, working together to find relationships.",
        "technical_content": r"""
<p>For each token embedding \( x \) (512-dim):</p>
<h4>Query (Q): "What am I looking for?"</h4>
<p>$$ Q = x \times W_Q \quad (512 \times 64 \rightarrow 64\text{-dim}) $$</p>

<h4>Key (K): "What do I contain?"</h4>
<p>$$ K = x \times W_K \quad (512 \times 64 \rightarrow 64\text{-dim}) $$</p>

<h4>Value (V): "What information do I carry?"</h4>
<p>$$ V = x \times W_V \quad (512 \times 64 \rightarrow 64\text{-dim}) $$</p>

<p><strong>Why 64?</strong> \( d_k = d_{model} / num\_heads = 512 / 8 = 64 \)</p>
<h4>Intuition:</h4>
<ul>
<li><strong>Query:</strong> "I'm the word 'sat', I need to know WHO sat"</li>
<li><strong>Key:</strong> "I'm the word 'cat', I describe a WHO"</li>
<li><strong>Value:</strong> "I'm 'cat', here's my semantic meaning to pass along"</li>
</ul>
""",
        "image_prompt": "Three superhero characters emerging from a single word, labeled Q, K, V, each with distinct color scheme (blue, red, green), showing transformation arrows from original word, dynamic action poses, comic book superhero style"
    },
    {
        "chapter": "3",
        "page_number": 6,
        "title": "The Attention Score Calculation",
        "narrative": "Q and K shake hands to determine compatibility.",
        "technical_content": r"""
<h3>Step 1: Compute Compatibility</h3>
<p>$$ \text{Attention Score} = \frac{Q \cdot K^T}{\sqrt{d_k}} $$</p>
<h4>Example for "sat" attending to all words:</h4>
<ul>
<li>\( Q_{sat} \cdot K_{The} = 2.3 \)</li>
<li>\( Q_{sat} \cdot K_{cat} = 8.7 \) &larr; <strong>High score! Related!</strong></li>
<li>\( Q_{sat} \cdot K_{sat} = 3.1 \)</li>
</ul>
<h3>Step 2: Scale by \( \sqrt{d_k} = \sqrt{64} = 8 \)</h3>
<p><strong>Scaled Scores:</strong> [0.29, 1.09, 0.39]</p>
<h4>Why divide by \( \sqrt{d_k} \)?</h4>
<ul>
<li>Prevents scores from getting too large</li>
<li>Keeps gradients stable</li>
<li>Critical for training!</li>
</ul>
<pre><code class="language-python"># Matrix Form (for sequence length n=3):
#       [The] [cat] [sat]
# [The] 0.29  0.45  0.31
# [cat] 0.35  0.88  0.41
# [sat] 0.29  1.09  0.39
</code></pre>
""",
        "image_prompt": "Comic panel showing word-characters shaking hands, energy bolts between them with floating numbers, scoreboard in background showing compatibility scores, action lines indicating strength of connection, vibrant energy effects"
    },
    {
        "chapter": "3",
        "page_number": 7,
        "title": "Softmax - Converting to Probabilities",
        "narrative": "Raw scores transformed into a probability distribution.",
        "technical_content": r"""
<h3>Softmax Function:</h3>
<p>$$ \text{attention\_weights} = \text{softmax}(\text{scores}) = \frac{\exp(\text{scores})}{\sum \exp(\text{scores})} $$</p>

<h4>For "sat" row: [0.29, 1.09, 0.39]</h4>
<p>\( \exp([0.29, 1.09, 0.39]) = [1.34, 2.97, 1.48] \)</p>
<p>Sum = 5.79</p>

<p><strong>Softmax:</strong> [1.34/5.79, 2.97/5.79, 1.48/5.79] = [0.23, 0.51, 0.26]</p>

<h4>Interpretation:</h4>
<ul>
<li>23% attention to "The"</li>
<li>51% attention to "cat" &larr; <strong>Most relevant!</strong></li>
<li>26% attention to "sat" itself</li>
</ul>
""",
        "image_prompt": "Comic panel showing a pie chart emerging from number scores, word-characters sized proportionally to their attention weights, largest character glowing brightest, probability percentages floating above, clean technical illustration style"
    },
    {
        "chapter": "3",
        "page_number": 8,
        "title": "Weighted Sum - Gathering Information",
        "narrative": "Values combined according to attention weights to create enriched representation.",
        "technical_content": r"""
<h3>Output = Attention_weights &times; Values</h3>
<p>For "sat":</p>
<p>$$ output_{sat} = 0.23 \times V_{The} + 0.51 \times V_{cat} + 0.26 \times V_{sat} $$</p>

<p>If \( V_{The} = [0.1, 0.2, ..., 0.3] \) (64-dim)<br>
\( V_{cat} = [0.8, 0.9, ..., 0.7] \) (64-dim)<br>
\( V_{sat} = [0.3, 0.4, ..., 0.2] \) (64-dim)</p>

<p>\( output_{sat} = [0.51, 0.58, ..., 0.49] \) (64-dim)</p>

<p><strong>Now "sat" has learned it's related to "cat"!</strong></p>
<h4>Complete Attention Formula:</h4>
<p>$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V $$</p>
""",
        "image_prompt": "Comic panel showing three translucent ghost-like value characters merging into one solid character, weighted by size/opacity, mathematical formula glowing in background, fusion energy effects, dramatic transformation scene"
    },
    {
        "chapter": "4",
        "page_number": 9,
        "title": "Eight Heads Are Better Than One",
        "narrative": "The transformer splits into 8 parallel versions, each focusing on different aspects.",
        "technical_content": r"""
<h3>Why 8 heads?</h3>
<ul>
<li>Head 1 might learn: syntactic relationships (subject-verb)</li>
<li>Head 2 might learn: semantic similarity</li>
<li>Head 3 might learn: positional proximity</li>
<li>Head 4 might learn: co-reference (pronouns)</li>
<li>... and so on</li>
</ul>

<p>Each head: \( d_k = d_v = 512/8 = 64 \) dimensions</p>

<h4>Parallel Processing:</h4>
<pre><code class="language-python">head_1 = Attention(Q_1, K_1, V_1)  # (n x 64)
head_2 = Attention(Q_2, K_2, V_2)  # (n x 64)
# ...
head_8 = Attention(Q_8, K_8, V_8)  # (n x 64)
</code></pre>

<p><strong>Concatenate:</strong> [head_1 | head_2 | ... | head_8] &rarr; (n &times; 512)</p>
<p><strong>Final Projection:</strong> Concat &times; \( W_O \) &rarr; (n &times; 512)</p>
""",
        "image_prompt": "Comic splash page with 8 identical characters in circle formation, each with different colored aura, viewing same scene from different angles, spider-web of connections between all characters, kaleidoscopic effect, multi-perspective composition"
    },
    {
        "chapter": "4",
        "page_number": 10,
        "title": "What Each Head Learns",
        "narrative": "Visualization of different attention patterns discovered by different heads.",
        "technical_content": r"""
<h3>Example Sentence: "The cat sat on the mat because it was tired"</h3>

<h4>Head 1 - Syntactic (Subject-Verb):</h4>
<p>"cat" &rarr; "sat" (strong)</p>
<p>"it" &rarr; "was" (strong)</p>

<h4>Head 2 - Long-range Dependencies:</h4>
<p>"it" &rarr; "cat" (pronoun resolution)</p>

<h4>Head 3 - Local Context:</h4>
<p>"sat" &rarr; "on" (adjacent words)</p>
<p>"on" &rarr; "the" (adjacent words)</p>

<h4>Head 4 - Semantic:</h4>
<p>"cat" &rarr; "tired" (living things get tired)</p>
<p>Attention Pattern Matrices for different heads shown side-by-side</p>
""",
        "image_prompt": "Comic panel grid showing 4 mini-scenes, each depicting different types of word connections with different colored energy beams, syntactic connections shown as rigid lines, semantic as flowing curves, local as tight spirals, diverse visual metaphors"
    },
    {
        "chapter": "5",
        "page_number": 11,
        "title": "The Power-Up Station",
        "narrative": "Each token visits a power-up station independently (position-wise).",
        "technical_content": r"""
<h3>Feed-Forward Network (FFN):</h3>
<p>$$ \text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2 $$</p>

<h4>Layer dimensions:</h4>
<ul>
<li>\( W_1 \): 512 &times; 2048 (expansion)</li>
<li>\( W_2 \): 2048 &times; 512 (projection back)</li>
</ul>

<h4>Why 2048?</h4>
<ul>
<li>4&times; expansion factor is standard</li>
<li>Gives network more capacity</li>
<li>Creates non-linear transformations</li>
</ul>

<h4>ReLU Activation:</h4>
<p>\( \max(0, x) \) - kills negative values. Introduces non-linearity - crucial for learning complex patterns!</p>

<p><strong>Applied to EACH token independently (unlike attention)</strong></p>
<p>Step-by-step for one token:</p>
<pre>
Input: [512-dim vector]
  &darr; x W_1 + b_1
Intermediate: [2048-dim vector]
  &darr; ReLU
Activated: [2048-dim vector] (some values zeroed)
  &darr; x W_2 + b_2
Output: [512-dim vector]
</pre>
""",
        "image_prompt": "Comic panel showing a word-character entering a power-up chamber, expanding into larger glowing form (2048), then compressing back down to original size but now glowing brighter, transformation sequence with energy effects, video game power-up aesthetic"
    },
    {
        "chapter": "5",
        "page_number": 12,
        "title": "Why FFN After Attention?",
        "narrative": "Attention gathers information, FFN processes it individually.",
        "technical_content": r"""
<h3>Division of Labor:</h3>

<h4>Multi-Head Attention:</h4>
<ul>
<li>Mixes information ACROSS tokens</li>
<li>"cat" learns from "sat", "the", etc.</li>
<li>Global information flow</li>
</ul>

<h4>Feed-Forward Network:</h4>
<ul>
<li>Processes EACH token independently</li>
<li>Refines the combined information</li>
<li>Local transformation</li>
</ul>

<h3>Together:</h3>
<ol>
<li>Attention: "Gather context from other words"</li>
<li>FFN: "Now think deeply about what I've learned"</li>
</ol>
<p><strong>Analogy:</strong> Attention = Group discussion, FFN = Individual reflection</p>
""",
        "image_prompt": "Two-panel comic: Panel 1 shows characters in circle sharing information with crossed energy beams (attention), Panel 2 shows same characters each in individual meditation pose with internal glow (FFN), contrasting collaborative vs individual processing"
    },
    {
        "chapter": "6",
        "page_number": 13,
        "title": "The Skip Connection Highway",
        "narrative": "Information has a shortcut path to preserve original signal.",
        "technical_content": r"""
<h3>Problem: Deep networks lose information through layers</h3>

<h4>Residual Connection:</h4>
<p>$$ \text{output} = \text{LayerNorm}(x + \text{Sublayer}(x)) $$</p>

<p>Where Sublayer is either:</p>
<ul>
<li>Multi-Head Attention</li>
<li>Feed-Forward Network</li>
</ul>

<h4>Example:</h4>
<p>\( x_{input} = [0.5, 0.3, 0.8, ...] \) (512-dim)</p>
<p>\( \text{attention\_output} = [0.1, 0.2, 0.3, ...] \) (512-dim)</p>

<p><strong>With residual:</strong><br>
\( x_{next} = x_{input} + \text{attention\_output} = [0.6, 0.5, 1.1, ...] \)</p>

<h4>Why crucial?</h4>
<ul>
<li>Helps gradients flow during backprop</li>
<li>Prevents vanishing gradients</li>
<li>Allows training very deep networks (6, 12, 24+ layers)</li>
</ul>
""",
        "image_prompt": "Comic panel showing two paths - main winding road with obstacles (sublayer processing) and parallel highway (skip connection), both merging at end, arrows showing information flow, road/highway metaphor, clear path visualization"
    },
    {
        "chapter": "6",
        "page_number": 14,
        "title": "Layer Normalization - Keeping Things Stable",
        "narrative": "A balancing act to keep values in reasonable range.",
        "technical_content": r"""
<h3>Layer Normalization:</h3>
<p>$$ \text{LN}(x) = \gamma \times \frac{x - \mu}{\sigma} + \beta $$</p>

<p>For each token vector (512 values):<br>
\( \mu \) = mean of the 512 values<br>
\( \sigma \) = standard deviation of the 512 values<br>
\( \gamma, \beta \) = learned parameters (scale and shift)</p>

<h4>Example:</h4>
<p>\( x = [100, 200, 150, 50] \) (simplified to 4-dim)</p>
<p>\( \mu = (100 + 200 + 150 + 50) / 4 = 125 \)</p>
<p>\( \sigma = \sqrt{\text{variance}} \approx 55.9 \)</p>

<p><strong>Normalized:</strong> [(100-125)/55.9, (200-125)/55.9, (150-125)/55.9, (50-125)/55.9] = [-0.45, 1.34, 0.45, -1.34]</p>

<p>Then scale and shift with learned \( \gamma, \beta \)</p>
<h4>Why?</h4>
<ul>
<li>Prevents exploding/vanishing activations</li>
<li>Stabilizes training</li>
<li>Allows higher learning rates</li>
</ul>
""",
        "image_prompt": "Comic panel showing scattered numbers of wildly different sizes being normalized into similar-sized balanced figures on a scale, before/after comparison, balance beam metaphor with numbers as weights, equilibrium visualization"
    },
    {
        "chapter": "7",
        "page_number": 15,
        "title": "Assembling the Encoder",
        "narrative": "All components come together into one mighty encoder block.",
        "technical_content": r"""
<h3>Single Encoder Block:</h3>

<pre>
Input: x  (n x 512)
  &darr;
1. Multi-Head Attention
   attention_out = MultiHeadAttention(x, x, x)
  &darr;
2. Add & Norm
   x = LayerNorm(x + attention_out)
  &darr;
3. Feed-Forward Network
   ffn_out = FFN(x)
  &darr;
4. Add & Norm
   x = LayerNorm(x + ffn_out)
  &darr;
Output: x  (n x 512)
</pre>

<p><strong>Standard Transformer:</strong> Stack 6 identical encoder blocks</p>
<p><strong>GPT-3:</strong> 96 layers!</p>
<h4>Complete Architecture Diagram:</h4>
<p>[Input Embedding + Positional Encoding] &rarr; [Encoder 1] &rarr; [Encoder 2] &rarr; ... &rarr; [Encoder 6] &rarr; [Output]</p>
""",
        "image_prompt": "Detailed technical schematic comic page showing complete encoder block as a machine/factory, input tokens entering at bottom, passing through labeled stages (attention chamber, residual highways, FFN station, normalization gates), isometric technical drawing style, cutaway view showing internal processes"
    },
    {
        "chapter": "8",
        "page_number": 16,
        "title": "The Decoder's Special Power",
        "narrative": "Decoder can't peek into the future (masked attention).",
        "technical_content": r"""
<h3>Decoder Differences:</h3>

<h4>1. Masked Self-Attention</h4>
<ul>
<li>Can only attend to previous positions</li>
<li>Prevents "cheating" during training</li>
</ul>

<pre>
Mask Matrix (for 4 tokens):
       t0  t1  t2  t3
t0  [  0  -&infin;  -&infin;  -&infin; ]
t1  [  0   0  -&infin;  -&infin; ]
t2  [  0   0   0  -&infin; ]
t3  [  0   0   0   0 ]
</pre>
<p>After softmax, -&infin; becomes 0 (no attention)</p>

<h4>2. Encoder-Decoder Attention</h4>
<ul>
<li>Query from decoder</li>
<li>Key & Value from encoder</li>
<li>"What did the input say?"</li>
</ul>

<h4>3. Feed-Forward (same as encoder)</h4>
<p><strong>Autoregressive Generation:</strong> Input: "Hello" &rarr; Output: "world"</p>
<ol>
<li>Step 1: Generate "w" given "Hello"</li>
<li>Step 2: Generate "o" given "Hello w"</li>
<li>Step 3: Generate "r" given "Hello wo"</li>
<li>... and so on</li>
</ol>
""",
        "image_prompt": "Comic panel showing decoder-character with one-way glasses/visor looking only backward at previous tokens (shown as fading trail), future tokens shown as blurred/locked, mask visualization as a diagonal barrier, time-travel prevention metaphor"
    },
    {
        "chapter": "8",
        "page_number": 17,
        "title": "Encoder-Decoder Attention Bridge",
        "narrative": "Decoder asks questions to the encoder's knowledge.",
        "technical_content": r"""
<h3>Cross-Attention Mechanism:</h3>

<p><strong>Decoder:</strong> "I'm generating 'le' in French"<br>
Query: \( Q = \text{decoder\_state} \times W_Q \)</p>

<p><strong>Encoder:</strong> "I have info about 'the cat'"<br>
Key: \( K = \text{encoder\_output} \times W_K \)<br>
Value: \( V = \text{encoder\_output} \times W_V \)</p>

<h4>Attention Computation:</h4>
<p>$$ \text{attention} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V $$</p>

<h4>Example:</h4>
<ul>
<li>French "le" strongly attends to English "the"</li>
<li>French "chat" strongly attends to English "cat"</li>
</ul>
<p><strong>This creates alignment between input and output!</strong></p>
<pre>
Attention Pattern (Translation example):
English:  The  cat  sat  on   mat
            &darr;   &darr;    &darr;   &darr;    &darr;
French:   Le  chat  ...  ...  ...
          &uarr;   &uarr;
       Strong connections where semantically aligned
</pre>
""",
        "image_prompt": "Comic panel showing decoder character (left) sending question beam to encoder character (right), encoder holding knowledge book, information flowing back as answer beam, bridge of light between two towers metaphor, clear communication visualization"
    },
    {
        "chapter": "9",
        "page_number": 18,
        "title": "From Vectors to Words",
        "narrative": "Final transformation back to vocabulary.",
        "technical_content": r"""
<h3>Final Linear Layer + Softmax:</h3>

<pre>
Decoder output: (n x 512)
  &darr; x W_output (512 x vocab_size)
Logits: (n x 50,000)
  &darr; softmax
Probabilities: (n x 50,000)
</pre>

<h4>Example for next token prediction:</h4>
<pre>
[
  "the": 0.001,
  "cat": 0.542,  &larr; Highest! Predict "cat"
  "dog": 0.123,
  "sat": 0.089,
  ...
  (50,000 words total)
]
</pre>

<h4>Sampling Strategies:</h4>
<ol>
<li><strong>Greedy:</strong> Always pick highest probability</li>
<li><strong>Top-k:</strong> Sample from top k tokens</li>
<li><strong>Nucleus (top-p):</strong> Sample from smallest set summing to p</li>
<li><strong>Temperature:</strong> Control randomness</li>
</ol>
""",
        "image_prompt": "Comic panel showing final vector character entering a 'word printer' machine, probabilities shown as different sized word bubbles emerging, largest bubble highlighted as winner, lottery ball machine aesthetic with words instead of numbers"
    },
    {
        "chapter": "9",
        "page_number": 19,
        "title": "Training - Teacher Forcing",
        "narrative": "How the transformer learns from examples.",
        "technical_content": r"""
<h3>Training Process:</h3>

<p><strong>Input:</strong> "The cat sat on the mat"<br>
<strong>Target:</strong> "Le chat était assis sur le tapis"</p>

<h4>Teacher Forcing:</h4>
<ul>
<li>Feed entire target sequence at once</li>
<li>Use masking to prevent future peeking</li>
<li>Compare predictions to actual next tokens</li>
</ul>

<h4>Loss Calculation (Cross-Entropy):</h4>
<p>At position i, predict token i+1:<br>
predicted_probs = [0.001, 0.542, ...]<br>
actual_token = "chat" (ID: 3421)</p>
<p>$$ \text{Loss} = -\log(\text{predicted\_probs}[3421]) = -\log(0.542) = 0.612 $$</p>
<p><strong>Total Loss = Average over all positions</strong></p>

<h4>Gradient Descent:</h4>
<ul>
<li>Compute gradients \( \partial \text{Loss}/\partial W \) for all weights</li>
<li>Update: \( W_{new} = W_{old} - \text{learning\_rate} \times \text{gradient} \)</li>
<li>Repeat for millions of examples!</li>
</ul>

<h4>Optimization (Adam):</h4>
<ul>
<li>Adaptive learning rates</li>
<li>Momentum for stable updates</li>
<li>Learning rate scheduling (warmup + decay)</li>
</ul>
""",
        "image_prompt": "Comic panel showing teacher character pointing to target answer on blackboard, student transformer comparing its answer, error shown as red distance/gap between predicted and actual, adjustment arrows showing weight updates, classroom learning metaphor"
    },
    {
        "chapter": "10",
        "page_number": 20,
        "title": "The Transformer Revolution",
        "narrative": "Summary of why transformers changed everything.",
        "technical_content": r"""
<h3>Revolutionary Aspects:</h3>

<ol>
<li><strong>Parallelization:</strong> RNN: O(n) sequential steps vs Transformer: O(1) parallel steps &rarr; 100&times; faster training!</li>
<li><strong>Long-Range Dependencies:</strong> Direct connections &rarr; Better at long texts!</li>
<li><strong>Interpretability:</strong> Attention weights = explicit relationships &rarr; Can visualize what model learned!</li>
<li><strong>Scalability:</strong> Architecture scales to billions of parameters (GPT-3: 175B) &rarr; Emergent capabilities!</li>
<li><strong>Transfer Learning:</strong> Pre-train once, fine-tune for many tasks &rarr; Democratized NLP!</li>
</ol>

<h4>Applications:</h4>
<ul>
<li><strong>BERT:</strong> Understanding (search, Q&A)</li>
<li><strong>GPT:</strong> Generation (writing, coding)</li>
<li><strong>T5:</strong> Translation, summarization</li>
<li><strong>Vision Transformers:</strong> Image classification</li>
<li><strong>AlphaFold:</strong> Protein structure prediction</li>
</ul>
""",
        "image_prompt": "Epic final splash page showing transformer as superhero standing triumphant, background showing multiple application bubbles (text, images, molecules), before/after comparison of AI capabilities, celebratory heroic composition, inspirational technological advancement theme"
    },
    {
        "chapter": "10",
        "page_number": 21,
        "title": "Hyperparameters Cheatsheet",
        "narrative": "Quick reference for transformer configurations.",
        "technical_content": """
<h3>Standard Transformer (Vaswani et al., 2017):</h3>
<pre>
d_model = 512          # Embedding dimension
d_ff = 2048            # FFN hidden size
num_heads = 8          # Attention heads
num_layers = 6         # Encoder/Decoder layers
d_k = d_v = 64         # Per-head dimensions
dropout = 0.1          # Dropout rate
vocab_size = 37000     # BPE tokens
max_seq_len = 512      # Max sequence length
</pre>

<h3>GPT-2 (Small):</h3>
<pre>
d_model = 768
d_ff = 3072
num_heads = 12
num_layers = 12
vocab_size = 50257
Parameters: 117M
</pre>

<h3>BERT (Base):</h3>
<pre>
d_model = 768
d_ff = 3072
num_heads = 12
num_layers = 12
vocab_size = 30522
Parameters: 110M
</pre>

<h3>Computational Cost:</h3>
<p>Self-Attention: \\( O(n^2 \\times d) \\)<br>
FFN: \\( O(n \\times d^2) \\)<br>
where n = sequence length, d = d_model</p>
<p>Memory for Attention: \\( O(n^2 \\times \\text{num\\_layers}) \\)</p>
""",
        "image_prompt": "Technical reference sheet styled as comic page with organized stat boxes, numbers highlighted in different colors, small character icons representing different model sizes, clean infographic style, easy-to-scan layout"
    },
    {
        "chapter": "Appendix",
        "page_number": 22,
        "title": "The Math Behind Attention",
        "narrative": "Complete Derivations.",
        "technical_content": r"""
<h3>Scaled Dot-Product Attention:</h3>
<p>$$ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$</p>

<h4>Why \( \sqrt{d_k} \) scaling?</h4>
<p>Dot product \( Q \cdot K \) has variance proportional to \( d_k \). Without scaling, softmax saturates.</p>

<p><strong>Proof:</strong><br>
If \( Q, K \sim N(0,1) \), then \( Q \cdot K \sim N(0, d_k) \)<br>
Dividing by \( \sqrt{d_k} \): \( Q \cdot K/\sqrt{d_k} \sim N(0,1) \)<br>
Keeps variance constant regardless of dimension!</p>

<h4>Multi-Head Concatenation:</h4>
<p>$$ \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O $$</p>
<p>where \( \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \)</p>

<h4>Parameter Count Analysis:</h4>
<ul>
<li>Per head: \( 3 \times (d_{model} \times d_k) = 3 \times 512 \times 64 = 98,304 \)</li>
<li>Total QKV: \( 8 \text{ heads} \times 98,304 = 786,432 \)</li>
<li>Output projection \( W^O \): \( 512 \times 512 = 262,144 \)</li>
<li>Total attention params: ~1M per layer</li>
</ul>
""",
        "image_prompt": "Technical diagram page with mathematical equations, Greek symbols, matrix visualizations, proof steps with arrows, academic paper aesthetic with comic book color scheme, clean mathematical typography"
    },
    {
        "chapter": "Appendix",
        "page_number": 23,
        "title": "Implementation Tricks & Optimizations",
        "narrative": "Technical Content.",
        "technical_content": r"""
<h3>1. Masking Implementation:</h3>
<p>attention_scores + mask_matrix<br>
where mask = -1e9 (effectively -&infin;)</p>

<h3>2. Efficient Matrix Multiplication:</h3>
<p>Batch all heads: (batch &times; heads &times; seq &times; d_k)<br>
Single matrix multiply instead of loops</p>

<h3>3. Flash Attention (2022):</h3>
<p>Tiling technique to avoid materializing full attention matrix<br>
Memory: O(n) instead of O(n²)<br>
Speed: 2-4&times; faster</p>

<h3>4. Position Encoding Alternatives:</h3>
<ul>
<li>Learned positional embeddings</li>
<li>Rotary Position Embedding (RoPE)</li>
<li>ALiBi (Attention with Linear Biases)</li>
</ul>

<h3>5. Activation Functions:</h3>
<ul>
<li>Original: ReLU</li>
<li>Modern: GELU, SwiGLU</li>
<li>GELU: \( x \cdot \Phi(x) \) where \( \Phi \) is Gaussian CDF</li>
<li>Smoother gradients, better performance</li>
</ul>

<h3>6. Attention Variants:</h3>
<ul>
<li>Sparse Attention: Only attend to subset</li>
<li>Local Attention: Fixed window</li>
<li>Longformer: Sliding window + global tokens</li>
<li>BigBird: Random + window + global</li>
</ul>
""",
        "image_prompt": "Comic page with multiple small technical diagrams, code snippets in speech bubbles, optimization shown as 'before/after' speed comparisons, engineering blueprint aesthetic, practical implementation focus"
    }
]
