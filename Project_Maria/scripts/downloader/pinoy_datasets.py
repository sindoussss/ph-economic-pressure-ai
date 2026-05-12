import os, random, datetime, json

class TagalogDatasetCreator:
    """Create and manage Taglish/Filipino datasets for training"""

    def __init__(self, dataset_dir="tagalog_datasets"):
        self.dataset_dir = os.path.join(BASE_DIR, dataset_dir)
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Core datasets
        self.datasets = {
            'conversational_pairs': [],
            'sentence_patterns': [],
            'taglish_switches': [],
            'modern_slang': [],
            'cultural_context': []
        }

        # Initialize with REAL Filipino data
        self._initialize_base_datasets()

    def _initialize_base_datasets(self):
        """Initialize with REAL Filipino conversation data"""

        # REAL FILIPINO CONVERSATION PATTERNS
        self.datasets['conversational_pairs'] = [
            # Greetings & Responses
            {"english": "How are you?", "tagalog": "Kamusta ka na?", "taglish": "Mustaaa, besh?"},
            {"english": "I'm good, thanks!", "tagalog": "Mabuti naman, salamat!",
             "taglish": "Ok naman, besh! Salamat!"},
            {"english": "What's up?", "tagalog": "Anong balita?", "taglish": "Anong meron? Musta?"},

            # Common questions
            {"english": "What do you want to do?", "tagalog": "Anong gusto mong gawin?",
             "taglish": "Anong trip mo gawin?"},
            {"english": "Can you help me?", "tagalog": "Pwede mo ba akong tulungan?",
             "taglish": "Pwede patulong, besh?"},
            {"english": "Do you understand?", "tagalog": "Naiintindihan mo ba?", "taglish": "Gets mo ba?"},

            # Tech talk (Taglish style)
            {"english": "I need to debug this code", "tagalog": "Kailangan kong i-debug ang kodigong ito",
             "taglish": "Need ko i-debug 'tong code, besh"},
            {"english": "The API is not working", "tagalog": "Hindi gumagana ang API",
             "taglish": "Di gumagana yung API, sakit sa ulo"},
            {"english": "Let me check the documentation", "tagalog": "Hayaan mo akong tingnan ang dokumentasyon",
             "taglish": "Wait lang, check ko docs"},
        ]

        # MODERN TAGLISH SENTENCE PATTERNS
        self.datasets['sentence_patterns'] = [
            # Pattern: [English structure] → [Natural Taglish]
            "I think {idea} → Feeling ko {idea}",
            "Maybe we should {action} → Baka pwede nating {action}",
            "Actually, {statement} → Sa totoo lang, {statement}",
            "I don't know about {topic} → Di ako sure sa {topic}",
            "Let me try to {action} → Subukan kong {action}",
            "That's really {adjective} → Ang {adjective} talaga niyan",
            "Can you explain {concept}? → Pwede mo ba i-explain yung {concept}?",
            "I want to learn {skill} → Gusto kong matuto ng {skill}",
            "This is too {adjective} → Ang {adjective} naman nito",
            "Thank you for {action} → Salamat sa {action}"
        ]

        # NATURAL TAGLISH SWITCH POINTS
        self.datasets['taglish_switches'] = [
            # When Filipinos naturally switch to English
            {"filipino_part": "Grabe, ", "english_switch": "I'm so tired", "context": "exhaustion"},
            {"filipino_part": "Ang hirap", "english_switch": "debug this code", "context": "frustration"},
            {"filipino_part": "Sobrang", "english_switch": "awesome ng result", "context": "excitement"},
            {"filipino_part": "Di ko", "english_switch": "understand this part", "context": "confusion"},
            {"filipino_part": "Try mo", "english_switch": "to restart the app", "context": "suggestion"},
        ]

        # 2024 FILIPINO INTERNET SLANG
        self.datasets['modern_slang'] = [
            {"slang": "Sheesh", "meaning": "impressed/surprised", "example": "Sheesh! Ang galing!",
             "context": "positive reaction"},
            {"slang": "Char/Charot", "meaning": "just kidding", "example": "Ang pogi mo! Charot!", "context": "joking"},
            {"slang": "Sana all", "meaning": "wish I had that too", "example": "May bagong phone ka? Sana all!",
             "context": "envy"},
            {"slang": "Edi wow", "meaning": "sarcastic wow", "example": "Naka-iPhone ka? Edi wow.",
             "context": "sarcasm"},
            {"slang": "Lodi", "meaning": "idol (reversed)", "example": "Ang galing mo mag-code, lodi!",
             "context": "admiration"},
            {"slang": "Petmalu", "meaning": "amazing (reversed lumetpat)", "example": "Petmalu 'tong code mo!",
             "context": "praise"},
            {"slang": "Wers", "meaning": "where is? (swer)", "example": "Wers yung file?", "context": "question"},
            {"slang": "Awit", "meaning": "that's sad/tough", "example": "Na-delete yung code? Awit.",
             "context": "sympathy"},
        ]

    def collect_from_conversation(self, english_text: str, taglish_response: str):
        """Collect data from actual conversations"""
        pair = {
            "english": english_text,
            "taglish": taglish_response,
            "timestamp": datetime.now().isoformat(),
            "word_count": len(taglish_response.split()),
            "taglish_ratio": self._calculate_taglish_ratio(taglish_response)
        }

        self.datasets['conversational_pairs'].append(pair)

        # Auto-save
        self._save_dataset('conversational_pairs')

        print(f"📝 Collected new Taglish pair: {len(self.datasets['conversational_pairs'])} total")

    def _calculate_taglish_ratio(self, text: str) -> float:
        """Calculate how Taglish a text is (0=all English, 1=all Filipino)"""
        # Simple word detection
        filipino_words = ['ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila',
                          'ang', 'ng', 'sa', 'ni', 'kay', 'para',
                          'at', 'o', 'pero', 'kasi', 'kaya', 'para',
                          'po', 'opo', 'ho', 'oho', 'salamat', 'kamusta']

        words = text.lower().split()
        if not words:
            return 0.5

        filipino_count = sum(1 for word in words if word in filipino_words)
        return filipino_count / len(words)

    def generate_training_examples(self, count: int = 100):
        """Generate training examples for fine-tuning"""
        examples = []

        for _ in range(count):
            # Pick random pattern
            pattern = random.choice(self.datasets['conversational_pairs'])

            example = {
                "input": pattern["english"],
                "output": pattern.get("taglish", pattern.get("tagalog", "")),
                "metadata": {
                    "taglish_ratio": pattern.get("taglish_ratio", 0.5),
                    "word_count": pattern.get("word_count", 0),
                    "source": "collected_conversation"
                }
            }

            # Add variations
            if random.random() < 0.3:
                example["output"] = self._add_filipino_flavor(example["output"])

            examples.append(example)

        return examples

    def _add_filipino_flavor(self, text: str) -> str:
        """Add natural Filipino flavor to text"""
        import random

        # Add filler words
        filler_words = ['eh', 'kasi', 'naman', 'din', 'lang']
        if random.random() < 0.4:
            words = text.split()
            if len(words) > 3:
                insert_pos = random.randint(1, len(words) - 1)
                words.insert(insert_pos, random.choice(filler_words))
                text = ' '.join(words)

        # Add Filipino endings
        endings = [' di ba?', ' no?', ' ah!', ' ha!', ' besh!', ' teh!']
        if text[-1] in '.!?' and random.random() < 0.3:
            text = text.rstrip('.!?') + random.choice(endings)

        return text

    def _save_dataset(self, dataset_name: str):
        """Save dataset to file"""
        file_path = os.path.join(self.dataset_dir, f"{dataset_name}.json")
        save_json(file_path, self.datasets[dataset_name])

    def load_all_datasets(self):
        """Load all datasets from files"""
        for dataset_name in self.datasets.keys():
            file_path = os.path.join(self.dataset_dir, f"{dataset_name}.json")
            if os.path.exists(file_path):
                self.datasets[dataset_name] = safe_load_json(file_path, [])

    def export_for_finetuning(self, format: str = "alpaca"):
        """Export datasets in training format"""
        training_data = []

        # Convert conversational pairs to training format
        for pair in self.datasets['conversational_pairs']:
            if 'english' in pair and 'taglish' in pair:
                if format == "alpaca":
                    training_data.append({
                        "instruction": "Respond in natural Taglish (Filipino-English mix)",
                        "input": pair['english'],
                        "output": pair['taglish']
                    })
                elif format == "chatml":
                    training_data.append({
                        "messages": [
                            {"role": "user", "content": pair['english']},
                            {"role": "assistant", "content": pair['taglish']}
                        ]
                    })

        # Add sentence patterns
        for pattern in self.datasets['sentence_patterns']:
            if '→' in pattern:
                eng, taglish = pattern.split('→', 1)
                training_data.append({
                    "instruction": "Convert to natural Taglish",
                    "input": eng.strip(),
                    "output": taglish.strip()
                })

        return training_data


class TaglishDataCollector:
    """Collect real Taglish data from Maria's conversations"""

    def __init__(self):
        self.conversation_log = []
        self.collection_file = os.path.join(BASE_DIR, "taglish_collections.json")

        # Load existing collections
        self.load_collections()

    def load_collections(self):
        """Load collected conversations"""
        if os.path.exists(self.collection_file):
            try:
                with open(self.collection_file, 'r', encoding='utf-8') as f:
                    self.conversation_log = json.load(f)
                print(f"📊 Loaded {len(self.conversation_log)} Taglish conversations")
            except:
                self.conversation_log = []

    def log_conversation(self, user_input: str, ai_response: str):
        """Log a conversation for dataset building"""
        # Detect if response contains Taglish/Filipino
        if self._contains_filipino(ai_response):
            entry = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "ai_response": ai_response,
                "language": self._detect_language_mix(ai_response),
                "word_count": len(ai_response.split()),
                "filipino_word_count": self._count_filipino_words(ai_response)
            }

            self.conversation_log.append(entry)

            # Keep only last 1000 entries
            if len(self.conversation_log) > 1000:
                self.conversation_log = self.conversation_log[-1000:]

            # Auto-save periodically
            if len(self.conversation_log) % 10 == 0:
                self.save_collections()

    def _contains_filipino(self, text: str) -> bool:
        """Check if text contains Filipino words"""
        filipino_indicators = [
            'po', 'opo', 'salamat', 'kamusta', 'ako', 'ikaw',
            'siya', 'kami', 'kayo', 'sila', 'ang', 'ng', 'sa'
        ]

        text_lower = text.lower()
        return any(word in text_lower for word in filipino_indicators)

    def _detect_language_mix(self, text: str) -> str:
        """Detect language mix in text"""
        words = text.lower().split()
        if not words:
            return "unknown"

        filipino_words = self._count_filipino_words(text)
        ratio = filipino_words / len(words)

        if ratio > 0.7:
            return "mostly_filipino"
        elif ratio > 0.3:
            return "taglish_mix"
        else:
            return "mostly_english"

    def _count_filipino_words(self, text: str) -> int:
        """Count Filipino words in text"""
        common_filipino = [
            'ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila',
            'ang', 'ng', 'sa', 'ni', 'kay', 'para',
            'at', 'o', 'pero', 'kasi', 'kaya',
            'po', 'opo', 'ho', 'oho',
            'salamat', 'kamusta', 'kumusta',
            'oo', 'hindi', 'bakit', 'paano', 'saan', 'kailan',
            'ito', 'iyan', 'iyon', 'dito', 'doon', 'diyan',
            'may', 'wala', 'meron', 'walang',
            'na', 'pa', 'din', 'rin', 'lang', 'naman',
            'ba', 'yata', 'pala', 'kaya', 'talaga',
            'mabuti', 'masama', 'maganda', 'pangit',
            'malaki', 'maliit', 'matanda', 'bata'
        ]

        words = text.lower().split()
        return sum(1 for word in words if word in common_filipino)

    def save_collections(self):
        """Save collected conversations"""
        save_json(self.collection_file, self.conversation_log)
        print(f"💾 Saved {len(self.conversation_log)} Taglish conversations")

    def get_training_data(self, min_filipino_words: int = 2):
        """Get high-quality Taglish training data"""
        training_pairs = []

        for entry in self.conversation_log:
            if entry['filipino_word_count'] >= min_filipino_words:
                training_pairs.append({
                    "input": entry['user_input'],
                    "output": entry['ai_response'],
                    "metadata": {
                        "language_mix": entry['language'],
                        "filipino_words": entry['filipino_word_count'],
                        "timestamp": entry['timestamp']
                    }
                })

        return training_pairs