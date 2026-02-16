# src/parent_assistant.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
import re
from datetime import datetime
from typing import List, Dict, Optional

class ParentAssistant:
    def __init__(self, index_path="./faiss_index"):
        """Initialize the parenting assistant"""
        print("🧘‍♀️ Starting Parent Assistant...")
        
        # Load model and database
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_database(index_path)
        
        # Emotional state patterns
        self.emotional_patterns = {
            "overwhelmed": [
                r"can't take", r"exhausted", r"too much", r"breaking point",
                r"losing it", r"can't handle", r"drowning"
            ],
            "angry": [
                r"furious", r"so mad", r"pissed", r"hate", r"livid",
                r"rage", r"want to scream"
            ],
            "guilty": [
                r"bad parent", r"failed", r"should have", r"shouldn't have",
                r"what's wrong with me", r"terrible mother", r"terrible father"
            ],
            "frustrated": [
                r"won't listen", r"keeps doing", r"stop", r"every time",
                r"always", r"never listens", r"tried everything"
            ],
            "sad": [
                r"sad", r"depressed", r"hopeless", r"alone", r"lonely",
                r"crying", r"heartbroken"
            ]
        }
        
        print("✅ Assistant ready")
    
    def load_database(self, index_path):
        """Load the FAISS database"""
        index_path = Path(index_path)
        index_file = index_path / "index.faiss"
        metadata_file = index_path / "metadata.pkl"
        
        if index_file.exists() and metadata_file.exists():
            self.index = faiss.read_index(str(index_file))
            with open(metadata_file, 'rb') as f:
                self.documents, self.metadata = pickle.load(f)
            print(f"✅ Loaded {len(self.documents)} parenting insights")
        else:
            print("⚠️ No database found. Please run embeddings_faiss.py first")
            self.index = None
            self.documents = []
            self.metadata = []
    
    def detect_emotion(self, text: str) -> Dict:
        """Detect parent's emotional state from input"""
        text_lower = text.lower()
        emotions_found = []
        
        for emotion, patterns in self.emotional_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    emotions_found.append(emotion)
                    break
        
        # If no specific emotion, check for general distress
        if not emotions_found:
            # Look for question marks, exclamation marks, or urgent words
            if "?" in text or "!" in text:
                emotions_found = ["seeking_help"]
            else:
                emotions_found = ["neutral"]
        
        return {
            "primary_emotion": emotions_found[0] if emotions_found else "neutral",
            "all_emotions": emotions_found,
            "urgency": "high" if any(word in text_lower for word in ["help", "emergency", "now", "quick"]) else "normal"
        }
    
    def get_relevant_wisdom(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant Montessori/parenting insights"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "relevance_score": float(distances[0][i])
                })
        
        return results
    
    def generate_response(self, user_input: str) -> Dict:
        """Generate a therapeutic parent-first response"""
        
        # Step 1: Detect parent's emotional state
        emotion_state = self.detect_emotion(user_input)
        
        # Step 2: Find relevant Montessori insights
        insights = self.get_relevant_wisdom(user_input, k=3)
        
        # Step 3: Extract child's age if mentioned
        age_match = re.search(r'(\d+)(?:\s*-\s*\d+)?\s*(year|month|yo|yr)', user_input.lower())
        child_age = age_match.group(0) if age_match else "unknown"
        
        # Step 4: Build therapeutic response
        response = {
            "timestamp": datetime.now().isoformat(),
            "parent_input": user_input,
            "emotional_state": emotion_state,
            "child_age": child_age,
            "response_phases": [],
            "sources": []
        }
        
        # PHASE 1: Parent Regulation (Always First!)
        regulation_phase = {
            "phase": "1️⃣ Parent First - Let's Ground You",
            "message": self._get_regulation_message(emotion_state)
        }
        response["response_phases"].append(regulation_phase)
        
        # PHASE 2: Child Understanding
        child_phase = {
            "phase": "2️⃣ Understanding Your Child",
            "message": self._get_child_understanding(insights, user_input)
        }
        response["response_phases"].append(child_phase)
        
        # PHASE 3: Montessori Technique
        technique_phase = {
            "phase": "3️⃣ Gentle Approach to Try",
            "message": self._get_technique(insights, emotion_state)
        }
        response["response_phases"].append(technique_phase)
        
        # PHASE 4: Trauma-Aware Reflection (gentle)
        if emotion_state["primary_emotion"] in ["angry", "guilty", "overwhelmed"]:
            trauma_phase = {
                "phase": "4️⃣ A Gentle Reflection",
                "message": self._get_trauma_reflection(emotion_state)
            }
            response["response_phases"].append(trauma_phase)
        
        # Add sources
        for insight in insights[:2]:  # Show top 2 sources
            response["sources"].append({
                "title": insight["metadata"]["title"],
                "excerpt": insight["text"][:100] + "..."
            })
        
        return response
    
    def _get_regulation_message(self, emotion_state: Dict) -> str:
        """Generate parent validation and regulation message"""
        emotion = emotion_state["primary_emotion"]
        
        messages = {
            "overwhelmed": (
                "It's completely understandable to feel overwhelmed. "
                "Parenting is the most demanding job in the world, and you're doing it 24/7. "
                "Take a slow breath with me... breathe in... breathe out. "
                "You're not alone in this feeling."
            ),
            "angry": (
                "Your anger is valid. It's okay to feel angry. "
                "You're not a bad parent for having these feelings. "
                "Before we address what's happening, let's take a moment. "
                "Your feelings matter, and they're telling you something important."
            ),
            "guilty": (
                "I hear how hard you're being on yourself. "
                "Wanting to be a good parent and feeling like you're falling short "
                "is one of the heaviest burdens to carry. "
                "You reached out for help - that's what good parents do."
            ),
            "frustrated": (
                "Frustration makes sense when you've tried everything. "
                "It's exhausting when nothing seems to work. "
                "Let's pause together for a moment. You're doing the best you can."
            ),
            "sad": (
                "I hear the sadness in your words. "
                "Parenting can bring up so much, including our own unprocessed feelings. "
                "It's okay to not be okay. You're safe here."
            ),
            "neutral": (
                "Thank you for reaching out. "
                "Whatever you're feeling right now is welcome here. "
                "Let's explore this together."
            )
        }
        
        return messages.get(emotion, messages["neutral"])
    
    def _get_child_understanding(self, insights: List[Dict], user_input: str) -> str:
        """Explain child's developmental perspective"""
        if not insights:
            return (
                "Every child's behavior is a form of communication. "
                "They're not giving you a hard time - they're HAVING a hard time. "
                "Their brain is still developing, and they need your calm to find their own."
            )
        
        # Use the most relevant insight
        top_insight = insights[0]["text"][:200]
        return (
            f"From a developmental perspective:\n\n"
            f"{top_insight}\n\n"
            f"This is normal for their age. They're not manipulating you - "
            f"they're showing you what they need."
        )
    
    def _get_technique(self, insights: List[Dict], emotion_state: Dict) -> str:
        """Suggest a gentle parenting technique"""
        if not insights:
            return (
                "Try this: Get down to their eye level, speak softly, "
                "and say 'I see you're having a hard time. I'm here with you.' "
                "Connection before correction."
            )
        
        # Use insight to inform technique
        return (
            f"Montessori approach:\n"
            f"1. Observe what your child is trying to communicate\n"
            f"2. Offer a choice within limits\n"
            f"3. Model the calm you want to see\n\n"
            f"Remember: You're not trying to control them - you're guiding them."
        )
    
    def _get_trauma_reflection(self, emotion_state: Dict) -> str:
        """Gentle trauma-aware reflection"""
        return (
            "Sometimes our strongest reactions come from our own childhood. "
            "When you feel this way, you might be tapping into how you were parented. "
            "That's not your fault. You're breaking cycles just by being here, "
            "trying to understand and do better. Be gentle with yourself."
        )
    
    def chat(self):
        """Simple chat interface"""
        print("\n" + "="*60)
        print("🧘‍♀️ MONTESSORI PARENT ASSISTANT")
        print("="*60)
        print("\nI'm here to support YOU first, then help with your child.")
        print("Type 'quit' to exit.\n")
        
        while True:
            print("\n" + "-"*60)
            user_input = input("💭 What's happening? > ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n✨ Take care of yourself. You're doing important work.\n")
                break
            
            if not user_input.strip():
                continue
            
            print("\n🤔 Thinking...\n")
            response = self.generate_response(user_input)
            
            # Display response phases
            for phase in response["response_phases"]:
                print(f"\n{phase['phase']}")
                print(phase['message'])
                print()
            
            # Show sources
            if response["sources"]:
                print("📚 Based on:")
                for source in response["sources"]:
                    print(f"  • {source['title']}")
            
            print("\n" + "="*60)

if __name__ == "__main__":
    assistant = ParentAssistant()
    assistant.chat()