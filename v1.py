import pygame
import random
import time
import json # For logging
from datetime import datetime, timedelta
import math # For distance calculations
import ollama # Import the ollama library
import numpy as np # For vector operations (cosine similarity)
import os # For file operations

# --- Pygame Initialization ---
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 900
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Town Simulation (Advanced Cognitive & World State)")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
BROWN = (139, 69, 19)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
DARK_GREY = (100, 100, 100)

# Fonts
FONT = pygame.font.Font(None, 24)
BIG_FONT = pygame.font.Font(None, 36)
SMALL_FONT = pygame.font.Font(None, 18)

# --- Game Parameters ---
FPS = 30
GAME_SPEED_MULTIPLIER = 0.005 # Extremely slow for many LLM calls
GAME_MINUTES_PER_HOUR = 60
GAME_HOURS_PER_DAY = 24
GAME_DAYS_TO_RUN = 1 # Keep very short for testing
SIMULATION_TOTAL_MINUTES = GAME_DAYS_TO_RUN * GAME_HOURS_PER_DAY * GAME_MINUTES_PER_HOUR

# Ollama Configuration
OLLAMA_MODEL = 'llama2'
OLLAMA_EMBEDDING_MODEL = 'nomic-embed-text'
REFLECTION_IMPORTANCE_THRESHOLD = 150

# File for persisting object states
OBJECT_STATE_FILE = "world_objects_state.json"

# Global simulation log and object registry
SIMULATION_LOG = []
WORLD_OBJECTS = {} # To store Object instances

# Global message box
MESSAGE_BOX_MESSAGES = []
MESSAGE_BOX_TIMER = 0
MESSAGE_BOX_DURATION_FRAMES = 5 * FPS

def show_message_box(message, color=BLACK):
    """Displays a temporary message on the screen."""
    global MESSAGE_BOX_MESSAGES, MESSAGE_BOX_TIMER
    MESSAGE_BOX_MESSAGES.append((message, color))
    MESSAGE_BOX_TIMER = MESSAGE_BOX_DURATION_FRAMES
    if len(MESSAGE_BOX_MESSAGES) > 5:
        MESSAGE_BOX_MESSAGES = MESSAGE_BOX_MESSAGES[-5:]

# --- Map Definition ---
# Added 'provides_food', 'provides_rest' flags to locations
LOCATIONS = {
    "Handyman_Workshop": {'rect': pygame.Rect(100, 50, 150, 80), 'color': BROWN, 'parent': 'Town_North', 'objects': ['workbench', 'tool_rack', 'bed'], 'provides_rest': True},
    "Handyman_ResourceShop": {'rect': pygame.Rect(100, 140, 150, 80), 'color': ORANGE, 'parent': 'Town_North', 'objects': ['counter', 'shelves_wood', 'shelves_metal']},
    "Toolsmith_Workshop": {'rect': pygame.Rect(300, 50, 150, 80), 'color': BROWN, 'parent': 'Town_North', 'objects': ['forge', 'anvil', 'grindstone', 'bed'], 'provides_rest': True},
    "Toolsmith_Shop": {'rect': pygame.Rect(300, 140, 150, 80), 'color': ORANGE, 'parent': 'Town_North', 'objects': ['display_case', 'counter', 'tools_for_sale']},
    "Doctor_Clinic": {'rect': pygame.Rect(500, 50, 150, 80), 'color': LIGHT_BLUE, 'parent': 'Town_North', 'objects': ['exam_table', 'medicine_cabinet', 'desk', 'bed'], 'provides_rest': True},
    "Doctor_MedicineGarden": {'rect': pygame.Rect(500, 140, 150, 80), 'color': GREEN, 'parent': 'Town_North', 'objects': ['herb_patch_1', 'herb_patch_2', 'watering_can']},
    "Mayor_Building": {'rect': pygame.Rect(100, 300, 150, 100), 'color': PURPLE, 'parent': 'Town_West', 'objects': ['mayor_desk', 'town_records', 'meeting_table', 'bed'], 'provides_rest': True},
    "Common_Space": {'rect': pygame.Rect(300, 250, 400, 200), 'color': GREY, 'parent': 'Town_Center', 'objects': ['bench_1', 'fountain', 'notice_board'], 'provides_rest': True},
    "Garbage_Processing_NonBio": {'rect': pygame.Rect(750, 250, 100, 80), 'color': DARK_GREY, 'parent': 'Town_East', 'objects': ['recycling_bins', 'compactor']},
    "Garbage_Processing_Bio": {'rect': pygame.Rect(750, 340, 100, 80), 'color': DARK_GREY, 'parent': 'Town_East', 'objects': ['compost_bins']},
    "Farmer_Building": {'rect': pygame.Rect(850, 600, 150, 80), 'color': BROWN, 'parent': 'Town_SouthEast', 'objects': ['farm_tools_storage', 'planning_desk', 'kitchen_stove', 'refrigerator', 'bed'], 'provides_rest': True, 'provides_food': True}, # Farmer has a kitchen
    "Farmer_Granary": {'rect': pygame.Rect(850, 690, 150, 80), 'color': YELLOW, 'parent': 'Town_SouthEast', 'objects': ['grain_silos', 'storage_bins']},
    "Farmer_Shop": {'rect': pygame.Rect(700, 600, 100, 80), 'color': ORANGE, 'parent': 'Town_SouthEast', 'objects': ['produce_stand', 'cash_register'], 'provides_food': True},
    "Crop_Field_1": {'rect': pygame.Rect(250, 600, 150, 100), 'color': DARK_GREEN, 'parent': 'Town_SouthWest', 'objects': ['scarecrow', 'irrigation_ditch_1']},
    "Crop_Field_2": {'rect': pygame.Rect(450, 600, 150, 100), 'color': DARK_GREEN, 'parent': 'Town_SouthWest', 'objects': ['tool_shed', 'irrigation_ditch_2']},
    "Crop_Processing_1": {'rect': pygame.Rect(250, 500, 150, 80), 'color': BROWN, 'parent': 'Town_SouthWest', 'objects': ['threshing_floor', 'sorting_table']},
    "Crop_Processing_2": {'rect': pygame.Rect(450, 500, 150, 80), 'color': BROWN, 'parent': 'Town_SouthWest', 'objects': ['drying_racks', 'packing_station']},
    "River_Access": {'rect': pygame.Rect(400, 700, 50, 100), 'color': LIGHT_BLUE, 'parent': 'Town_South', 'objects': ['fishing_spot', 'water_pump']},
    "River": {'rect': pygame.Rect(0, 800, SCREEN_WIDTH, 100), 'color': LIGHT_BLUE, 'parent': 'Town_South', 'objects': ['river_bank', 'flowing_water']},
    "Town_Shop_1": {'rect': pygame.Rect(200, 450, 80, 50), 'color': ORANGE, 'parent': 'Town_Center', 'objects': ['general_goods_shelf', 'shop_counter'], 'provides_food': True},
    "Town_Shop_2": {'rect': pygame.Rect(600, 450, 80, 50), 'color': ORANGE, 'parent': 'Town_Center', 'objects': ['clothing_rack', 'book_shelf']},

    "Town_North": {'rect': pygame.Rect(0, 0, SCREEN_WIDTH, 240), 'color': (240,240,240), 'parent': 'World', 'objects': []},
    "Town_West": {'rect': pygame.Rect(0, 240, 290, 160), 'color': (240,240,240), 'parent': 'World', 'objects': []},
    "Town_Center": {'rect': pygame.Rect(290, 240, 420, 260), 'color': (240,240,240), 'parent': 'World', 'objects': []},
    "Town_East": {'rect': pygame.Rect(710, 240, SCREEN_WIDTH-710, 260), 'color': (240,240,240), 'parent': 'World', 'objects': []},
    "Town_SouthWest": {'rect': pygame.Rect(0, 490, 690, 300), 'color': (240,240,240), 'parent': 'World', 'objects': []},
    "Town_SouthEast": {'rect': pygame.Rect(690, 490, SCREEN_WIDTH-690, 300), 'color': (240,240,240), 'parent': 'World', 'objects': []},
    "Town_South": {'rect': pygame.Rect(0, 790, SCREEN_WIDTH, SCREEN_HEIGHT-790), 'color': (240,240,240), 'parent': 'World', 'objects': []},
    "World": {'rect': pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 'color': (255,255,255), 'parent': None, 'objects': []}
}

# --- Object Class Definition ---
class WorldObject:
    def __init__(self, name, location_name, initial_state="default", properties=None, can_be_used_by_multiple_agents=False):
        self.id = f"{location_name}_{name}"
        self.name = name
        self.location_name = location_name
        self.current_state = initial_state
        self.properties = properties if properties else {} # e.g., {'is_on': False, 'food_count': 10}
        self.can_be_used_by_multiple_agents = can_be_used_by_multiple_agents
        self.current_user = None # Agent object using this, if exclusive

    def get_description(self):
        props_str = ", ".join([f"{k}: {v}" for k,v in self.properties.items()])
        return f"Object '{self.name}' at '{self.location_name}' is currently '{self.current_state}'. Properties: {{{props_str}}}"

    def to_dict(self): # For saving
        return {
            'id': self.id, 'name': self.name, 'location_name': self.location_name,
            'current_state': self.current_state, 'properties': self.properties,
            'can_be_used_by_multiple_agents': self.can_be_used_by_multiple_agents,
            'current_user_name': self.current_user.name if self.current_user else None
        }

    @classmethod
    def from_dict(cls, data, agents_dict=None): # For loading, now accepts agents_dict
        obj = cls(data['name'], data['location_name'], data['current_state'], data.get('properties',{}), data.get('can_be_used_by_multiple_agents', False))
        obj.id = data['id']
        if data.get('current_user_name') and agents_dict:
            obj.current_user = agents_dict.get(data['current_user_name'])
        return obj

def initialize_world_objects(agents_dict=None): # Now accepts agents_dict for resolving users
    global WORLD_OBJECTS
    if os.path.exists(OBJECT_STATE_FILE):
        try:
            with open(OBJECT_STATE_FILE, 'r') as f:
                objects_data = json.load(f)
                WORLD_OBJECTS = {obj_id: WorldObject.from_dict(data, agents_dict) for obj_id, data in objects_data.items()}
                print(f"Loaded {len(WORLD_OBJECTS)} object states from {OBJECT_STATE_FILE}")
                return
        except Exception as e:
            print(f"Error loading object states: {e}. Initializing fresh.")
            WORLD_OBJECTS = {} # Reset if error

    # If no file or error, initialize from LOCATIONS
    for loc_name, loc_data in LOCATIONS.items():
        # Add location name to its own data
        LOCATIONS[loc_name]['name'] = loc_name
        if loc_data.get('objects'):
            for obj_name in loc_data['objects']:
                obj_id = f"{loc_name}_{obj_name}"
                props = {}
                can_multiple = False
                if "stove" in obj_name: props = {'is_on': False, 'is_burning': False, 'cook_progress': 0}
                if "refrigerator" in obj_name: props = {'food_count': 10, 'max_food': 10}
                if "bed" in obj_name: props = {'is_occupied': False}
                if "bench" in obj_name or "table" in obj_name: can_multiple = True
                
                WORLD_OBJECTS[obj_id] = WorldObject(obj_name, loc_name, properties=props, can_be_used_by_multiple_agents=can_multiple)
    print(f"Initialized {len(WORLD_OBJECTS)} fresh object states.")


def save_world_objects():
    with open(OBJECT_STATE_FILE, 'w') as f:
        serializable_objects = {obj_id: obj.to_dict() for obj_id, obj in WORLD_OBJECTS.items()}
        json.dump(serializable_objects, f, indent=2)
    # print(f"Saved {len(WORLD_OBJECTS)} object states to {OBJECT_STATE_FILE}")

# --- Game Time Management ---
current_datetime = datetime(2023, 2, 13, 7, 0, 0) # Start at 7 AM, Feb 13, 2023
game_day, game_hour, game_minute = 0,0,0 # Will be updated

def get_current_game_time_as_datetime():
    return current_datetime

def advance_game_time(minutes=1):
    global current_datetime, game_day, game_hour, game_minute
    current_datetime += timedelta(minutes=minutes)
    game_day = current_datetime.day - 12 # Assuming Feb 13 is Day 1
    game_hour = current_datetime.hour
    game_minute = current_datetime.minute

# --- Ollama Integration Functions ---
def _call_ollama(prompt: str, agent_name: str = "Agent") -> str:
    """Makes a call to the local Ollama server for text generation."""
    try:
        # print(f"\n--- Ollama Prompt for {agent_name} ---\n{prompt}\n--- End ---")
        response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, stream=False)
        # print(f"--- Ollama Resp for {agent_name} ---\n{response['response'].strip()}\n--- End ---")
        return response['response'].strip()
    except Exception as e:
        print(f"Error calling Ollama Gen for {agent_name}: {e}")
        show_message_box(f"Ollama Gen Error: {e}", RED)
        return _mock_ollama_response(prompt, agent_name) # Fallback

def _call_ollama_embedding(text: str, agent_name: str = "Agent") -> list[float]:
    """Makes a call to the local Ollama server for embeddings."""
    try:
        response = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error calling Ollama Embed for {agent_name}: {e}")
        show_message_box(f"Ollama Embed Error: {e}", RED)
        return [0.0] * 768 # Common embedding dimension for nomic-embed-text

def _mock_ollama_response(prompt: str, agent_name: str = "Agent") -> str:
    """ Fallback mock responses. Updated for new features. """
    if "rate the likely poignancy" in prompt: return str(random.randint(3, 7))
    if "Sketch out a plan for the day" in prompt:
        return "1. Morning routine. 2. Go to work and interact with objects. 3. Lunch break (fulfill hunger). 4. Continue work, considering relationships. 5. Evening relaxation, update emotional state."
    if "Decompose this high-level plan step" in prompt:
        if "interact with objects" in prompt: return "1. Walk to workbench. 2. Use workbench for 10 minutes. 3. Check tool_rack status."
        if "fulfill hunger" in prompt: return "1. Walk to Farmer_Shop. 2. Buy apple from produce_stand. 3. Eat apple."
        return "1. Initiate sub-task A. 2. Perform sub-task B. 3. Complete sub-task C."
    if "Should they react?" in prompt:
        return "No, continue current plan."
    if "interpreting the message" in prompt:
        return "Intent: The sender is sharing information. Tone: friendly."
    if "What are 3 most salient high-level questions" in prompt:
        return "1. How can I improve my skills today? 2. How are my relationships affecting my mood? 3. What object interaction would be most beneficial now?"
    if "What 5 high-level insights can you infer" in prompt:
        return "Insight: Agent is focused on skill development and social connections."
    if "What would" in prompt and "say next" in prompt:
        if "friendship_score" in prompt: return "It's great to see you, friend! How's your day?"
        return "Interesting. Tell me more."
    if "core characteristics" in prompt: return f"{agent_name} is a complex individual with evolving traits."
    if "current daily occupation" in prompt: return "Engaging in varied tasks and interactions."
    if "feeling about his recent progress" in prompt: return "Feeling content with personal growth and social bonds."
    if "What happens to the state of" in prompt: return "{\"agent_outcome\": \"Agent used the object successfully.\", \"object_new_state\": \"used\", \"object_property_changes\": {\"count\": 9}}"
    if "What is the new emotional state" in prompt: return "neutral"
    if "How should the relationship scores" in prompt: return "friendship_delta: +5; trust_delta: +2"
    if "What should" in prompt and "do to fulfill the need for" in prompt:
        if "hunger" in prompt: return "1. Go to Farmer_Shop. 2. Buy food from produce_stand. 3. Eat food."
        if "rest" in prompt: return "1. Go to Handyman_Workshop. 2. Use bed. 3. Sleep for 60 minutes."
        return "1. Identify resource. 2. Go to resource. 3. Use resource."
    return "Ollama mock response: Processed with new logic."

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if vec1.shape != vec2.shape: return 0.0
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot_product / (norm_a * norm_b)

# --- New Ollama Call Functions for Enhanced Sophistication ---
def call_ollama_for_emotional_update(agent_name: str, current_emotion: str, recent_events_summary: str) -> str:
    prompt = (
        f"Agent {agent_name}'s current emotional state is '{current_emotion}'.\n"
        f"Recent significant events for {agent_name}:\n{recent_events_summary}\n"
        f"Based on these events, what is {agent_name}'s new emotional state? (Choose from: neutral, happy, sad, angry, surprised, anxious, content). Respond with only the emotional state."
    )
    return _call_ollama(prompt, agent_name).lower()

def call_ollama_for_relationship_update(agent_name_1: str, agent_name_2: str, interaction_summary: str, current_friendship: int, current_trust: int) -> dict:
    prompt = (
        f"Agent {agent_name_1} and Agent {agent_name_2} just had an interaction summarized as: '{interaction_summary}'.\n"
        f"Their current friendship score (0-100) is {current_friendship}, and trust score (0-100) is {current_trust}.\n"
        f"How should these scores change? Provide deltas (e.g., friendship_delta: +5, trust_delta: -2). Respond with only 'friendship_delta: [value]; trust_delta: [value]'."
    )
    response = _call_ollama(prompt, agent_name_1)
    deltas = {'friendship_delta': 0, 'trust_delta': 0}
    try:
        parts = response.split(';')
        for part in parts:
            key_value = part.split(':')
            if len(key_value) == 2:
                key = key_value[0].strip()
                value = float(key_value[1].strip().replace('+', '')) # Handle + sign
                if key == 'friendship_delta': deltas['friendship_delta'] = value
                elif key == 'trust_delta': deltas['trust_delta'] = value
    except Exception as e:
        print(f"Error parsing relationship update: {e} from response: {response}")
    return deltas

def call_ollama_for_need_fulfillment_plan(agent_name: str, need_type: str, agent_summary:str, current_location: str, known_locations_info: str, dt_obj:datetime) -> list[str]:
    prompt = (
        f"Agent: {agent_name}\nSummary: {agent_summary}\nCurrently at: {current_location}\nTime: {dt_obj.strftime('%A, %B %d, %Y, %I:%M %p')}\n"
        f"CRITICAL NEED: {need_type}.\n"
        f"Known locations relevant to this need:\n{known_locations_info}\n"
        f"Generate a short, high-priority 2-3 step plan for {agent_name} to fulfill this need for '{need_type}'. Specify locations and objects. Respond as a numbered list."
    )
    plan_text = _call_ollama(prompt, agent_name)
    return [step.strip() for step in plan_text.split('\n') if step.strip() and (step[0].isdigit() or step[0] == '-')]

def call_ollama_for_object_interaction_outcome(agent_name: str, action_description: str, obj: WorldObject, agent_summary: str) -> dict:
    prompt = (
        f"Agent: {agent_name}\nSummary: {agent_summary}\n"
        f"Action: '{action_description}'\n"
        f"Object: '{obj.name}' at '{obj.location_name}'\n"
        f"Object's current state: '{obj.current_state}'\n"
        f"Object's current properties: {json.dumps(obj.properties)}\n"
        f"What is the outcome for the agent (e.g., 'Agent is now busy cooking', 'Agent obtained wood') AND "
        f"what is the new state of '{obj.name}' (e.g., 'on', 'empty') AND "
        f"how do its properties change (e.g., food_count: 9, is_on: true)?\n"
        f"Respond ONLY in JSON format: {{\"agent_outcome\": \"text\", \"object_new_state\": \"text\", \"object_property_changes\": {{\"key\": \"value\", ...}}}}"
    )
    response_str = _call_ollama(prompt, agent_name)
    try:
        outcome = json.loads(response_str)
        return {
            "agent_outcome": outcome.get("agent_outcome", f"Agent used {obj.name}."),
            "object_new_state": outcome.get("object_new_state", obj.current_state),
            "object_property_changes": outcome.get("object_property_changes", {})
        }
    except json.JSONDecodeError:
        print(f"Error decoding JSON from LLM for object interaction: {response_str}")
        return {"agent_outcome": f"Agent used {obj.name}.", "object_new_state": obj.current_state, "object_property_changes": {}}

# --- Agent Class (Modified for Sophistication) ---
class Agent:
    def __init__(self, name, role, description, start_location_name, color, all_agent_names):
        self.name = name
        self.role = role
        self.initial_description = description
        self.color = color
        self.current_location_name = start_location_name
        self.x, self.y = LOCATIONS[start_location_name]['rect'].center
        self.size = 20
        self.memory_stream = []
        self.high_level_plan = []
        self.detailed_plan = []
        self.current_high_level_action_index = 0
        self.current_detailed_action_index = 0
        self.needs = {'hunger': 0, 'rest': 0, 'social': 0, 'sickness': 0, 'fulfillment': 0}
        self.status = "idle"
        self.target_location_name = start_location_name
        self.target_x, self.target_y = self.x, self.y
        self.current_message = ""
        self.message_timer = 0
        self.goals = self._get_initial_goals()
        self.dialogue_history = []
        self.cached_summary = ""
        self.last_summary_update_day = -1
        self.previous_day_activity_summary = "No activities recorded yet for the previous day."
        self.busy_with_object_id = None
        self.busy_timer = 0

        # Enhanced Sophistication Attributes
        self.emotional_state = "neutral"
        self.relationships = {name: {'friendship_score': 20, 'trust_score': 20, 'last_interaction_time': None} for name in all_agent_names if name != self.name} # Default 20/100

        seed_memories = self.initial_description.split(';')
        for mem_text in seed_memories:
            if mem_text.strip():
                self.add_memory(f"{self.name} {mem_text.strip()}", "Seed", importance_score=9,
                                location_context=self.current_location_name,
                                dt_obj=get_current_game_time_as_datetime() - timedelta(days=1, minutes=random.randint(1,1440)))
        self.update_cached_summary()

    def _get_initial_goals(self):
        base_goals = []
        if self.role == "Handyman": base_goals = ["Maintain town infrastructure", "Fix broken items", "Acquire resources from shops"]
        elif self.role == "Toolsmith": base_goals = ["Craft high-quality tools", "Sell tools to townspeople", "Innovate tool designs"]
        elif self.role == "Doctor": base_goals = ["Heal sick agents", "Gather medicinal herbs", "Promote town health and hygiene"]
        elif self.role == "Mayor": base_goals = ["Oversee town affairs", "Organize community events", "Resolve inter-agent disputes", "Ensure town prosperity"]
        elif self.role == "Farmer": base_goals = ["Grow abundant crops", "Efficiently process food", "Sell produce to townspeople", "Manage farm resources and irrigation"]
        else: base_goals = ["Survive", "Interact with environment"]
        return base_goals + ["Seek personal fulfillment", "Maintain good relationships"]

    def add_memory(self, description, memory_type, importance_score=None, related_agents=None, location_context=None, objects_involved=None, dt_obj=None):
        dt_obj = dt_obj or get_current_game_time_as_datetime()
        if importance_score is None:
            importance_score = call_ollama_for_importance_score(description, self.name)
        
        embedding = _call_ollama_embedding(description, self.name)

        memory = {
            'description': description, 'embedding': embedding,
            'creation_timestamp_obj': dt_obj, 'last_accessed_timestamp_obj': dt_obj,
            'type': memory_type, 'importance_score': importance_score,
            'recency_score': 1.0, 'relevance_score': 0.0,
            'related_agents': related_agents if related_agents else [],
            'location_context': location_context if location_context else self.current_location_name,
            'objects_involved': objects_involved if objects_involved else []
        }
        self.memory_stream.append(memory)
        SIMULATION_LOG.append({
            'timestamp': datetime.now().isoformat(),
            'game_time': dt_obj.strftime("%Y-%m-%d %H:%M:%S"),
            'agent': self.name,
            'type': f"Memory_Added_{memory_type}",
            'details': {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in memory.items() if k != 'embedding'}
        })

    def update_recency_scores(self, query_dt: datetime):
        """Decays recency scores for all memories based on time since last access."""
        for memory in self.memory_stream:
            hours_since_last_access = (query_dt - memory['last_accessed_timestamp_obj']).total_seconds() / 3600.0
            memory['recency_score'] = math.exp(-0.01 * hours_since_last_access) # Decay factor 0.01 per hour

    def retrieve_memories(self, query: str, count: int = 10, query_dt: datetime = None) -> list[dict]:
        query_dt = query_dt or get_current_game_time_as_datetime()
        self.update_recency_scores(query_dt) # Decay recency before retrieval

        query_embedding = _call_ollama_embedding(query, self.name) # Generate embedding for the query

        scored_memories = []
        for memory in self.memory_stream:
            relevance = cosine_similarity(query_embedding, memory['embedding'])
            memory['relevance_score'] = relevance # Update for sorting

            combined_score = memory['recency_score'] + \
                             (memory['importance_score'] / 10.0) + \
                             memory['relevance_score']
            scored_memories.append((combined_score, memory))
        
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        retrieved_mem_list = []
        for score, mem in scored_memories[:count]:
            mem['last_accessed_timestamp_obj'] = query_dt # Update last access time for THIS retrieval
            retrieved_mem_list.append(mem)
            
        return retrieved_mem_list

    def update_cached_summary(self):
        """Updates the agent's cached summary as per GA Paper Appendix A."""
        current_dt = get_current_game_time_as_datetime()
        if current_dt.day == self.last_summary_update_day and self.cached_summary: 
            return

        core_char_mem_text = "\n".join([m['description'] for m in self.retrieve_memories(f"{self.name}'s core characteristics", count=5, query_dt=current_dt)])
        core_chars = call_ollama_for_agent_summary_component(self.name, f"{self.name}'s core characteristics", core_char_mem_text)
        occupation_mem_text = "\n".join([m['description'] for m in self.retrieve_memories(f"{self.name}'s current daily occupation", count=5, query_dt=current_dt)])
        occupation = call_ollama_for_agent_summary_component(self.name, f"{self.name}'s current daily occupation", occupation_mem_text)
        progress_mem_text = "\n".join([m['description'] for m in self.retrieve_memories(f"{self.name}'s feeling about their recent progress in life", count=5, query_dt=current_dt)])
        progress_feeling = call_ollama_for_agent_summary_component(self.name, f"{self.name}'s feeling about their recent progress in life", progress_mem_text)

        self.cached_summary = (f"{self.name}, the {self.role}. {self.initial_description.split(';')[0]}. "
                               f"Currently feeling {self.emotional_state}. Core: {core_chars}. "
                               f"Occupation: {occupation}. Progress: {progress_feeling}.")
        self.last_summary_update_day = current_dt.day
        self.add_memory(f"Updated cached summary: {self.cached_summary}", "Internal", importance_score=3, dt_obj=current_dt)

    def update_emotional_state(self):
        current_dt = get_current_game_time_as_datetime()
        recent_events = self.retrieve_memories("recent impactful events for emotional update", count=5, query_dt=current_dt)
        recent_events_summary = "; ".join([mem['description'] for mem in recent_events if mem['importance_score'] > 5])
        
        if recent_events_summary:
            new_emotion = call_ollama_for_emotional_update(self.name, self.emotional_state, recent_events_summary)
            if new_emotion != self.emotional_state:
                self.emotional_state = new_emotion
                self.add_memory(f"Emotional state changed to {self.emotional_state} due to recent events.", "EmotionalChange", importance_score=6, dt_obj=current_dt)
                show_message_box(f"{self.name} is now feeling {self.emotional_state}", self.color)

    def update_relationship(self, other_agent_name, interaction_summary):
        current_dt = get_current_game_time_as_datetime()
        if other_agent_name not in self.relationships:
            self.relationships[other_agent_name] = {'friendship_score': 20, 'trust_score': 20, 'last_interaction_time': current_dt}
        
        rel = self.relationships[other_agent_name]
        deltas = call_ollama_for_relationship_update(self.name, other_agent_name, interaction_summary, rel['friendship_score'], rel['trust_score'])
        
        rel['friendship_score'] = max(0, min(100, int(rel['friendship_score'] + deltas.get('friendship_delta',0))))
        rel['trust_score'] = max(0, min(100, int(rel['trust_score'] + deltas.get('trust_delta',0))))
        rel['last_interaction_time'] = current_dt
        
        self.add_memory(f"Relationship with {other_agent_name} updated after interaction '{interaction_summary}'. Friendship: {rel['friendship_score']:.0f}, Trust: {rel['trust_score']:.0f}.",
                        "RelationshipUpdate", importance_score=5, related_agents=[other_agent_name], dt_obj=current_dt)

    def reflect(self):
        """Agent reflects based on GA Paper Section 4.2."""
        current_dt = get_current_game_time_as_datetime()
        recent_memories = self.retrieve_memories("recent experiences for reflection", count=100, query_dt=current_dt) 
        
        if sum(m['importance_score'] for m in recent_memories[:20]) < REFLECTION_IMPORTANCE_THRESHOLD: 
            return

        recent_mem_descriptions_text = "\n".join([f"{idx+1}. {m['description']}" for idx, m in enumerate(recent_memories)])
        
        questions_to_reflect_on = call_ollama_for_reflection_questions(self.name, recent_mem_descriptions_text)
        if not questions_to_reflect_on:
            return

        show_message_box(f"{self.name} is reflecting deeply...", PURPLE)

        for question in questions_to_reflect_on:
            memories_for_question = self.retrieve_memories(question, count=15, query_dt=current_dt)
            memories_for_question_text = "\n".join([f"{idx+1}. {m['description']}" for idx, m in enumerate(memories_for_question)])
            
            insight = call_ollama_for_reflection_insights(self.name, question, memories_for_question_text)
            if insight and "Insight:" in insight:
                parsed_insight = insight.split("Insight:", 1)[1].strip()
                self.add_memory(f"Reflection on '{question}': {parsed_insight}", "Reflection", 
                                importance_score=call_ollama_for_importance_score(parsed_insight, self.name), dt_obj=current_dt)

    def plan_daily_activities(self):
        """Generates daily plan based on GA Paper Section 4.3."""
        current_dt = get_current_game_time_as_datetime()
        self.update_cached_summary() 
        
        yesterday_dt = current_dt - timedelta(days=1)
        yesterday_memories = [m['description'] for m in self.memory_stream 
                              if m['creation_timestamp_obj'].day == yesterday_dt.day and m['type'] == 'Observation']
        self.previous_day_activity_summary = "Yesterday was uneventful."
        if yesterday_memories:
            self.previous_day_activity_summary = "Key activities yesterday: " + "; ".join(random.sample(yesterday_memories, min(len(yesterday_memories), 5)))


        self.high_level_plan = call_ollama_for_planning(self.name, self.role, self.cached_summary, 
                                                        self.previous_day_activity_summary,
                                                        game_day, game_hour)
        self.current_high_level_action_index = 0
        self.detailed_plan = [] 
        self.current_detailed_action_index = 0
        self.add_memory(f"Planned daily activities: {'; '.join(self.high_level_plan)}", "Plan", importance_score=8, dt_obj=current_dt)
        self.status = "idle"
        if self.high_level_plan:
            self.decompose_current_plan_step() 

    def decompose_current_plan_step(self):
        """Decomposes high-level plan into detailed actions."""
        current_dt = get_current_game_time_as_datetime()
        if not self.high_level_plan or self.current_high_level_action_index >= len(self.high_level_plan):
            self.detailed_plan = []
            return

        high_level_step = self.high_level_plan[self.current_high_level_action_index]
        self.detailed_plan = call_ollama_for_decompose_plan_step(self.name, high_level_step, self.cached_summary, self.current_location_name, current_dt)
        self.current_detailed_action_index = 0
        self.add_memory(f"Decomposed '{high_level_step}' into: {'; '.join(self.detailed_plan)}", "PlanDetail", importance_score=6, dt_obj=current_dt)

    def react_to_observation(self, observation_text, observed_entity_name=None, observed_action_status=None):
        """Reacts to observation based on GA Paper Section 4.3.1."""
        current_dt = get_current_game_time_as_datetime()
        self.update_cached_summary()
        
        current_detailed_step = self.detailed_plan[self.current_detailed_action_index] if self.detailed_plan and self.current_detailed_action_index < len(self.detailed_plan) else "No specific detailed action."
        
        context_summary = ""
        if observed_entity_name: 
            relevant_mems = self.retrieve_memories(f"{self.name}'s relationship with {observed_entity_name} and {observed_entity_name}'s action of {observed_action_status}", count=5, query_dt=current_dt)
            relevant_mems_desc = [m['description'] for m in relevant_mems]
            rel_info = ""
            if observed_entity_name in self.relationships:
                rel = self.relationships[observed_entity_name]
                rel_info = f" Current relationship with {observed_entity_name}: Friendship {rel['friendship_score']:.0f}, Trust {rel['trust_score']:.0f}."
            
            base_context = call_ollama_for_reaction_context_summary(self.name, self.name, observed_entity_name, observed_action_status, relevant_mems_desc)
            context_summary = base_context + rel_info
        else: 
            context_summary = "This is an environmental observation."

        full_agent_summary_for_reaction = self.cached_summary + f" Currently feeling: {self.emotional_state}."

        should_react, reaction_description = call_ollama_for_reaction(self.name, full_agent_summary_for_reaction, current_detailed_step, observation_text, context_summary, current_dt)
        
        if should_react:
            self.add_memory(f"Reacted to '{observation_text}'. Reaction: {reaction_description}. Emotion: {self.emotional_state}", "Reaction", 
                            importance_score=9, related_agents=[observed_entity_name] if observed_entity_name else [], dt_obj=current_dt)
            
            if "New plan:" in reaction_description or "new plan:" in reaction_description:
                new_plan_str = reaction_description.split("plan:", 1)[1].strip()
                self.high_level_plan = [step.strip() for step in new_plan_str.split('.') if step.strip()] 
                self.current_high_level_action_index = 0
                self.detailed_plan = []
                self.current_detailed_action_index = 0
                if self.high_level_plan: self.decompose_current_plan_step()
                show_message_box(f"{self.name}: Plan changed: {self.high_level_plan[0] if self.high_level_plan else 'Cleared'}", ORANGE)
            elif "Go to" in reaction_description and any(loc in reaction_description for loc in LOCATIONS): 
                words = reaction_description.split()
                try:
                    target_loc_idx = words.index("to") + 1
                    target_loc_name_candidate = words[target_loc_idx].replace(".","").replace(",","").strip()
                    found_loc = None
                    for loc_key in LOCATIONS.keys():
                        if target_loc_name_candidate.lower() in loc_key.lower():
                            found_loc = loc_key
                            break
                    if found_loc:
                        self.high_level_plan = [f"Go to {found_loc}.", reaction_description.split(found_loc)[-1].strip()] 
                        self.current_high_level_action_index = 0
                        self.detailed_plan = []
                        self.current_detailed_action_index = 0
                        if self.high_level_plan: self.decompose_current_plan_step()
                        show_message_box(f"{self.name}: Reacting by going to {found_loc}", ORANGE)
                    else: 
                        self.current_message = f"Reaction: {reaction_description}"
                        self.message_timer = 3 * FPS
                except (ValueError, IndexError):
                    self.current_message = f"Reaction: {reaction_description}" 

            else: 
                self.current_message = f"Reaction: {reaction_description}" 
                self.message_timer = 3 * FPS
        else:
            self.add_memory(f"Observed '{observation_text}', decided not to react, continued current plan.", "Observation", 
                            importance_score=2, related_agents=[observed_entity_name] if observed_entity_name else [], dt_obj=current_dt)

    def communicate(self, target_agent, message_context_from_plan: str):
        """Agent sends a natural language message."""
        current_dt = get_current_game_time_as_datetime()
        self.update_cached_summary() 

        relationship_summary = "an acquaintance"
        if target_agent.name in self.relationships:
            rel = self.relationships[target_agent.name]
            relationship_summary = f"Relationship with {target_agent.name}: Friendship {rel['friendship_score']:.0f}, Trust {rel['trust_score']:.0f}."
        
        message_to_send = call_ollama_for_dialogue(self.name, target_agent.name, self.cached_summary, 
                                                  self.dialogue_history, message_context_from_plan,
                                                  relationship_summary, self.emotional_state)

        self.current_message = message_to_send
        self.message_timer = 3 * FPS 
        self.add_memory(f"Said to {target_agent.name}: '{message_to_send}' Emotion: {self.emotional_state}", "CommunicationSent", 
                        related_agents=[target_agent.name], location_context=self.current_location_name, dt_obj=current_dt)
        
        self.dialogue_history.append(f"{self.name} ({current_dt.strftime('%H:%M')}): {message_to_send}")
        target_agent.receive_communication(self, message_to_send, current_dt)
        
        self.update_relationship(target_agent.name, f"initiated conversation: '{message_to_send}'")


    def receive_communication(self, sender_agent, message: str, comm_dt: datetime):
        """Agent receives and interprets communication."""
        llm_interpretation_response = call_ollama_for_message_interpretation(sender_agent.name, self.name, message)
        
        interpretation = llm_interpretation_response
        tone = "neutral" 
        if "Intent:" in llm_interpretation_response and "Tone:" in llm_interpretation_response:
            parts = llm_interpretation_response.split("Tone:")
            interpretation = parts[0].replace("Intent:", "").strip()
            if len(parts) > 1:
                tone = parts[1].strip().lower()

        received_description = f"Received from {sender_agent.name}: '{message}'. Interpretation: {interpretation}. Tone: {tone}"
        self.add_memory(received_description, "CommunicationReceived", 
                        related_agents=[sender_agent.name], location_context=self.current_location_name, dt_obj=comm_dt)
        
        self.dialogue_history.append(f"{sender_agent.name} ({comm_dt.strftime('%H:%M')}): {message}")
        if len(self.dialogue_history) > 10: self.dialogue_history.pop(0) 

        self.react_to_observation(f"{sender_agent.name} said: '{message}' (meaning: {interpretation}, tone: {tone})", 
                                  observed_entity_name=sender_agent.name, observed_action_status=f"saying '{message}'")
        
        self.update_relationship(sender_agent.name, f"received message: '{message}' (Interpreted as: {interpretation}, Tone: {tone})")
        
        if tone in ["angry", "sad"] and self.emotional_state not in ["angry", "sad", "anxious"]: # Negative tone might affect emotion
            self.update_emotional_state()

    def move_to(self, location_name):
        if location_name in LOCATIONS:
            self.target_location_name = location_name
            self.target_x, self.target_y = LOCATIONS[location_name]['rect'].center
            self.status = "moving"
            self.add_memory(f"Started moving towards {location_name}.", "Observation", importance_score=3, location_context=self.current_location_name)
        else:
            show_message_box(f"Warning: {self.name} cannot find {location_name}!", RED)

    def update_position(self):
        if self.status == "moving":
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            distance = math.sqrt(dx**2 + dy**2)
            if distance > 5:
                move_speed = 5 
                self.x += dx / distance * move_speed
                self.y += dy / distance * move_speed
            else:
                self.x, self.y = self.target_x, self.target_y
                self.current_location_name = self.target_location_name
                self.add_memory(f"Arrived at {self.current_location_name}.", "Observation", importance_score=3, location_context=self.current_location_name)
                self.status = "idle" 
                if self.detailed_plan and self.current_detailed_action_index < len(self.detailed_plan):
                    action_text = self.detailed_plan[self.current_detailed_action_index].lower()
                    if ("walk to" in action_text or "go to" in action_text) and self.target_location_name.lower() in action_text :
                        self.current_detailed_action_index += 1


    def update_needs(self):
        self.needs['hunger'] += 0.01 
        self.needs['rest'] += 0.01
        self.needs['social'] += 0.005
        self.needs['fulfillment'] += 0.002 # Passive gain/loss

        if random.random() < 0.0001 and self.needs['sickness'] == 0:
            self.needs['sickness'] = random.randint(1, 10)
            self.add_memory(f"Started feeling sick (sickness level: {self.needs['sickness']}).", "Observation", importance_score=7)
            self.react_to_observation("I am feeling sick.")
            show_message_box(f"{self.name} is feeling sick!", RED)

        # Critical needs trigger address_critical_need
        if self.needs['hunger'] > 7 and self.status not in ["moving", "eating", "addressing_need"]:
            self.address_critical_need('hunger', agents)
        if self.needs['rest'] > 7 and self.status not in ["moving", "resting", "addressing_need"]:
            self.address_critical_need('rest', agents)
        if self.needs['social'] > 7 and self.status not in ["moving", "communicating", "addressing_need"]:
            self.address_critical_need('social', agents)
        if self.needs['fulfillment'] < 3 and self.status not in ["moving", "addressing_need"]: # Low fulfillment
            self.address_critical_need('fulfillment', agents)
        
        if self.needs['sickness'] > 0 and self.role != "Doctor":
            if self.current_location_name != "Doctor_Clinic" and self.status not in ["moving", "addressing_need"]:
                self.address_critical_need('sickness', agents) # Force going to doctor
            elif self.current_location_name == "Doctor_Clinic" and self.status == "idle":
                self.needs['sickness'] = max(0, self.needs['sickness'] - 0.5) 
                if self.needs['sickness'] == 0:
                    self.add_memory("Feeling better after treatment.", "Observation", importance_score=7)
                    show_message_box(f"{self.name} recovered!", GREEN)
                    self.plan_daily_activities() 
                else:
                    self.add_memory(f"Still feeling sick (sickness level: {self.needs['sickness']:.1f}).", "Observation", importance_score=5)
        elif self.needs['sickness'] > 0 and self.role == "Doctor":
            self.needs['sickness'] = max(0, self.needs['sickness'] - 0.2) 
            if self.needs['sickness'] == 0:
                self.add_memory("As a doctor, I've healed myself.", "Observation", importance_score=7)
                show_message_box(f"Doc healed themselves!", GREEN)

    def perceive_environment(self, all_agents):
        current_dt = get_current_game_time_as_datetime()
        for other_agent in all_agents:
            if other_agent.name != self.name and other_agent.current_location_name == self.current_location_name:
                obs_text = f"Saw {other_agent.name} (the {other_agent.role}) at {self.current_location_name}."
                self.add_memory(obs_text, "Observation", importance_score=1, related_agents=[other_agent.name], dt_obj=current_dt)
                if other_agent.current_message and other_agent.message_timer > 0: 
                    self.add_memory(f"Heard {other_agent.name} say: '{other_agent.current_message}'", "Observation", importance_score=3, related_agents=[other_agent.name], dt_obj=current_dt)
        
        location_data = LOCATIONS.get(self.current_location_name)
        if location_data and location_data.get('objects'):
            for obj_name in location_data['objects']: # Iterate through object names defined in LOCATIONS
                obj_id = f"{self.current_location_name}_{obj_name}"
                if obj_id in WORLD_OBJECTS:
                    world_obj = WORLD_OBJECTS[obj_id]
                    self.add_memory(world_obj.get_description(), "Observation", 
                                    importance_score=1, objects_involved=[world_obj.name], dt_obj=current_dt)
        else:
             self.add_memory(f"Observing the surroundings at {self.current_location_name}.", "Observation", importance_score=1, dt_obj=current_dt)


    def interact_with_object(self, obj_id: str, action_description: str):
        current_dt = get_current_game_time_as_datetime()
        if obj_id not in WORLD_OBJECTS:
            self.add_memory(f"Tried to interact with non-existent object '{obj_id}'.", "Error", dt_obj=current_dt)
            return

        obj = WORLD_OBJECTS[obj_id]

        if obj.current_user and obj.current_user != self and not obj.can_be_used_by_multiple_agents:
            self.add_memory(f"Tried to use '{obj.name}' but it's in use by {obj.current_user.name}.", "Observation", dt_obj=current_dt)
            self.status = "idle" # Cannot use, go idle or re-plan
            return

        self.update_cached_summary() # Ensure summary is fresh for LLM call
        interaction_result = call_ollama_for_object_interaction_outcome(self.name, action_description, obj, self.cached_summary)

        obj.current_state = interaction_result["object_new_state"]
        for prop_key, prop_value in interaction_result["object_property_changes"].items():
            # Basic type conversion for common cases (e.g., bool from string)
            if isinstance(obj.properties.get(prop_key), bool) and isinstance(prop_value, str):
                obj.properties[prop_key] = prop_value.lower() == 'true'
            elif isinstance(obj.properties.get(prop_key), (int, float)) and isinstance(prop_value, str):
                try: obj.properties[prop_key] = float(prop_value)
                except ValueError: obj.properties[prop_key] = prop_value
            else:
                obj.properties[prop_key] = prop_value

        if not obj.can_be_used_by_multiple_agents:
            obj.current_user = self
            self.busy_with_object_id = obj_id
            self.busy_timer = random.randint(5, 15) # Busy for 5-15 minutes
            self.status = f"using_{obj.name.replace(' ','_')}"
        
        self.add_memory(f"Interacted with '{obj.name}' ({action_description}). Agent outcome: {interaction_result['agent_outcome']}. Object now '{obj.current_state}', props {obj.properties}",
                        "ObjectInteraction", importance_score=5, objects_involved=[obj.name], dt_obj=current_dt)
        show_message_box(f"{self.name} used {obj.name}. New state: {obj.current_state}", CYAN)
        
        # Check if interaction fulfilled a need (e.g., eating, resting)
        if "food_count" in obj.properties and obj.properties["food_count"] < obj.properties.get("max_food", 10):
            self.needs['hunger'] = max(0, self.needs['hunger'] - 5)
            self.add_memory(f"Ate from {obj.name}, hunger reduced.", "NeedFulfilled", dt_obj=current_dt)
            self.needs['fulfillment'] += 1
        if "is_occupied" in obj.properties and obj.properties["is_occupied"] == True and "bed" in obj.name:
            self.needs['rest'] = max(0, self.needs['rest'] - 5)
            self.add_memory(f"Rested on {obj.name}, rest need reduced.", "NeedFulfilled", dt_obj=current_dt)
            self.needs['fulfillment'] += 1


    def perform_action(self, all_agents):
        current_dt = get_current_game_time_as_datetime()

        if self.busy_timer > 0:
            self.busy_timer -=1
            if self.busy_timer == 0 and self.busy_with_object_id:
                if self.busy_with_object_id in WORLD_OBJECTS:
                     WORLD_OBJECTS[self.busy_with_object_id].current_user = None
                self.add_memory(f"Finished using object {self.busy_with_object_id}.", "ObjectInteraction", dt_obj=current_dt)
                self.busy_with_object_id = None
                self.status = "idle"
            return # Agent is busy

        # Check critical needs first (already handled by address_critical_need in update_needs)
        # If agent is addressing a need, its plan will be set by address_critical_need

        if not self.detailed_plan or self.current_detailed_action_index >= len(self.detailed_plan):
            if self.current_high_level_action_index < len(self.high_level_plan):
                self.decompose_current_plan_step()
                if not self.detailed_plan: 
                    self.current_high_level_action_index += 1 
                    self.status = "idle"
                return 
            else: 
                self.status = "idle" 
                if game_hour < 2 or game_hour > 22: self.plan_daily_activities()
                return

        current_action_text = self.detailed_plan[self.current_detailed_action_index]
        action_completed_this_tick = True 
        action_verb = current_action_text.split(" ")[0].lower()
        
        # --- Action Execution Logic ---
        if "walk to" in current_action_text.lower() or "go to" in current_action_text.lower():
            try:
                target_loc_name = current_action_text.split(" to ", 1)[1].split(" ", 1)[0].replace(".","").replace(",","").strip()
                target_loc_key = None
                for loc_key in LOCATIONS: 
                    if target_loc_name.lower() in loc_key.lower():
                        target_loc_key = loc_key
                        break
                if target_loc_key:
                    if self.current_location_name != target_loc_key:
                        self.move_to(target_loc_key)
                        action_completed_this_tick = False 
                else: 
                    self.add_memory(f"Could not determine target location for: '{current_action_text}'. Skipping.", "Error", dt_obj=current_dt)
            except IndexError:
                 self.add_memory(f"Could not parse 'walk to' action: '{current_action_text}'. Skipping.", "Error", dt_obj=current_dt)

        elif "use " in current_action_text.lower() or "sleep" in current_action_text.lower() or "eat" in current_action_text.lower() or "buy" in current_action_text.lower() :
            # Try to extract object name from action text
            obj_name_in_action = ""
            if "use " in current_action_text.lower(): obj_name_in_action = current_action_text.lower().split("use ", 1)[1].split(" for ",1)[0].split(" to ",1)[0].strip()
            elif "sleep" in current_action_text.lower(): obj_name_in_action = "bed"
            elif "eat " in current_action_text.lower(): obj_name_in_action = "food" # Generic, LLM will map to specific object
            elif "buy " in current_action_text.lower(): obj_name_in_action = current_action_text.lower().split("buy ", 1)[1].split(" from ",1)[0].strip()
            
            found_obj_id = None
            for obj_id, world_obj_instance in WORLD_OBJECTS.items():
                if world_obj_instance.location_name == self.current_location_name and obj_name_in_action.replace(" ","_") in world_obj_instance.name.replace(" ","_"):
                    found_obj_id = obj_id
                    break
            
            if found_obj_id:
                self.interact_with_object(found_obj_id, current_action_text)
                if self.busy_timer > 0: action_completed_this_tick = False 
            else:
                self.add_memory(f"Could not find object '{obj_name_in_action}' for action '{current_action_text}' at {self.current_location_name}. Skipping.", "Error", dt_obj=current_dt)

        elif "discuss" in action_verb or "collaborate" in action_verb or "greet" in action_verb or \
             "ask" in action_verb or "talk" in action_verb or "explain" in action_verb or "inquire" in action_verb:
            self.status = "communicating"
            potential_targets = [a for a in all_agents if a.name != self.name and a.current_location_name == self.current_location_name]
            if potential_targets:
                target_agent = random.choice(potential_targets)
                self.communicate(target_agent, current_action_text) 
            else:
                self.add_memory(f"Wanted to '{current_action_text}', but no one is here at {self.current_location_name}.", "Observation", dt_obj=current_dt)
        
        elif any(verb in action_verb for verb in ["inspect", "check", "review", "prepare", "assess", "tend", "craft", "forge", "sharpen", "open", "arrange", "wait", "handle", "sell", "announce"]):
            self.status = "working"
            self.add_memory(f"Working: '{current_action_text}' at {self.current_location_name}.", "Work", importance_score=5, dt_obj=current_dt)
            self.needs['hunger'] = max(0, self.needs['hunger'] - 0.01)
            self.needs['rest'] = max(0, self.needs['rest'] - 0.01)

        elif "rest" in action_verb or "relax" in action_verb:
            if self.current_location_name == self.get_home_location():
                self.status = "resting"
                self.needs['rest'] = max(0, self.needs['rest'] - 0.5) 
                self.add_memory(f"Resting: '{current_action_text}' at home.", "Rest", importance_score=5, dt_obj=current_dt)
            else: 
                self.move_to(self.get_home_location())
                action_completed_this_tick = False
        
        elif "get treated" in current_action_text.lower() and self.current_location_name == "Doctor_Clinic":
            pass # Handled by update_needs

        elif any(verb in action_verb for verb in ["collect", "acquire", "gather", "purchase", "load", "fill", "transport", "unload", "process", "plan", "draft", "post"]):
            self.status = "task_oriented" 
            self.add_memory(f"Performing task: '{current_action_text}' at {self.current_location_name}.", "Task", importance_score=4, dt_obj=current_dt)
        else:
            self.status = "doing_something"
            self.add_memory(f"Performing unknown/generic action: '{current_action_text}'", "Observation", importance_score=3, dt_obj=current_dt)

        if action_completed_this_tick:
            self.current_detailed_action_index += 1

        if self.current_detailed_action_index >= len(self.detailed_plan):
            self.current_detailed_action_index = 0
            self.current_high_level_action_index += 1 
            self.detailed_plan = [] 
            self.add_memory("Completed all detailed actions for current high-level plan step.", "PlanStepCompletion", importance_score=4, dt_obj=current_dt)
            self.status = "idle" 
            if self.current_high_level_action_index >= len(self.high_level_plan):
                 self.add_memory("Completed all high-level plans for the day.", "PlanCompletion", importance_score=5, dt_obj=current_dt)
                 if game_hour < 6 or game_hour > 22: 
                    self.plan_daily_activities()

    def address_critical_need(self, need_type, all_agents):
        current_dt = get_current_game_time_as_datetime()
        self.update_cached_summary()
        self.add_memory(f"CRITICAL NEED: {need_type} is high ({self.needs[need_type]:.1f}). Seeking fulfillment.", "NeedCritical", dt_obj=current_dt)
        show_message_box(f"{self.name} has critical need: {need_type}!", RED)

        known_locations_info = ""
        relevant_locations = []
        if need_type == 'hunger':
            relevant_locations = [data['name'] for name, data in LOCATIONS.items() if data.get('provides_food')]
        elif need_type == 'rest':
            relevant_locations = [data['name'] for name, data in LOCATIONS.items() if data.get('provides_rest')]
        elif need_type == 'social':
            # Find agents with good relationship scores
            social_targets = [name for name, rel_data in self.relationships.items() if rel_data['friendship_score'] > 50]
            if social_targets:
                known_locations_info = f"Consider interacting with: {', '.join(social_targets)}. Common spaces might be useful."
            else:
                known_locations_info = "Common_Space might be a good place to find others."
            relevant_locations = [data['name'] for name,data in LOCATIONS.items() if 'Common_Space' in name or 'Shop' in name]
        elif need_type == 'sickness':
            relevant_locations = ["Doctor_Clinic"]
        elif need_type == 'fulfillment':
            # This is more abstract, might involve work or social interaction
            relevant_locations = [data['name'] for name,data in LOCATIONS.items() if 'Workshop' in name or 'Shop' in name or 'Common_Space' in name]

        if relevant_locations:
            known_locations_info += "\nKnown relevant locations: " + "; ".join(relevant_locations)

        fulfillment_plan_steps = call_ollama_for_need_fulfillment_plan(self.name, need_type, self.cached_summary, self.current_location_name, known_locations_info, current_dt)

        if fulfillment_plan_steps:
            self.add_memory(f"Generated urgent plan for {need_type}: {'; '.join(fulfillment_plan_steps)}", "UrgentPlan", dt_obj=current_dt)
            self.high_level_plan = fulfillment_plan_steps + self.high_level_plan[self.current_high_level_action_index:]
            self.current_high_level_action_index = 0
            self.detailed_plan = []
            self.current_detailed_action_index = 0
            self.status = "addressing_need"
            if self.high_level_plan: self.decompose_current_plan_step()
        else:
            self.add_memory(f"Could not generate fulfillment plan for {need_type}.", "Error", dt_obj=current_dt)
            self.status = "idle"

    def update(self, all_agents):
        self.update_needs()
        self.update_position() 

        if self.message_timer > 0:
            self.message_timer -= 1
            if self.message_timer == 0:
                self.current_message = "" 
                if self.dialogue_history and len(self.dialogue_history) > 0: 
                    if self.dialogue_history[-1].startswith(self.name):
                        if len(self.dialogue_history) > 2 : 
                             self.dialogue_history = []


        if get_current_game_time_as_datetime().minute % 30 == 0: 
            self.update_cached_summary()
        
        if get_current_game_time_as_datetime().minute % 10 == 0: 
            self.perceive_environment(all_agents)

        if get_current_game_time_as_datetime().minute % 15 == 0: 
            self.update_emotional_state()

        if get_current_game_time_as_datetime().minute == 5: 
            self.reflect()
        
        if self.status == "idle" and not self.detailed_plan and self.high_level_plan and self.current_high_level_action_index < len(self.high_level_plan):
            self.decompose_current_plan_step()

        if self.status != "moving" and self.busy_timer <= 0 : 
            self.perform_action(all_agents)
        elif self.busy_timer > 0: 
            self.busy_timer -= 1
            if self.busy_timer == 0 and self.busy_with_object_id:
                if self.busy_with_object_id in WORLD_OBJECTS:
                    WORLD_OBJECTS[self.busy_with_object_id].current_user = None
                self.add_memory(f"Finished using object {self.busy_with_object_id}.", "ObjectInteraction")
                self.busy_with_object_id = None
                self.status = "idle"

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)
        name_text = FONT.render(f"{self.name} ({self.emotional_state[0].upper()})", True, BLACK) 
        screen.blit(name_text, (self.x - name_text.get_width() / 2, self.y - self.size - 20))
        
        if self.current_message:
            words = self.current_message.split(' ')
            lines = []
            current_line = ""
            for word in words:
                if SMALL_FONT.size(current_line + " " + word)[0] < 150:
                    current_line += " " + word
                else:
                    lines.append(current_line.strip())
                    current_line = word
            lines.append(current_line.strip()) 

            max_line_width = 0
            total_height = 0
            for line in lines:
                line_surface = SMALL_FONT.render(line, True, BLACK)
                max_line_width = max(max_line_width, line_surface.get_width())
                total_height += line_surface.get_height()
            
            padding = 10
            bubble_width = max_line_width + 2 * padding
            bubble_height = total_height + 2 * padding
            bubble_rect = pygame.Rect(self.x - bubble_width / 2, self.y - self.size - bubble_height - 25 - (name_text.get_height() if name_text else 0), bubble_width, bubble_height) 
            
            pygame.draw.rect(screen, WHITE, bubble_rect, border_radius=5)
            pygame.draw.rect(screen, BLACK, bubble_rect, 2, border_radius=5) 

            text_y = bubble_rect.top + padding
            for line in lines:
                line_surface = SMALL_FONT.render(line, True, BLACK)
                screen.blit(line_surface, (bubble_rect.centerx - line_surface.get_width() / 2, text_y))
                text_y += line_surface.get_height()


# --- Simulation Setup ---
# Initialize agents first to get their names for object loading
agent_names_list = ["Handy", "Tooly", "Doc", "May", "Farmy"]
agents = [
    Agent("Handy", "Handyman", "Diligent worker; focused on town maintenance; repairs broken things; values practicality.", "Handyman_Workshop", RED, agent_names_list),
    Agent("Tooly", "Toolsmith", "Master craftsman; invents tools; helps community; detail-oriented.", "Toolsmith_Workshop", BLUE, agent_names_list),
    Agent("Doc", "Doctor", "Compassionate healer; dedicated to well-being; knowledgeable in herbs; promotes health.", "Doctor_Clinic", GREEN, agent_names_list),
    Agent("May", "Mayor", "Town leader; responsible for governance; organizes events; diplomatic.", "Mayor_Building", PURPLE, agent_names_list),
    Agent("Farmy", "Farmer", "Backbone of food supply; nurtures crops; manages resources; hardworking.", "Farmer_Building", YELLOW, agent_names_list),
]
agents_by_name = {agent.name: agent for agent in agents} # Create dict for quick lookup

initialize_world_objects(agents_by_name) # Now pass agents_by_name for resolving current_user

# Initial daily plan for all agents
for agent in agents:
    agent.plan_daily_activities()

# --- Game Loop ---
running = True
clock = pygame.time.Clock()

while running:
    current_dt_loop = get_current_game_time_as_datetime()
    game_day_sim_loop = current_dt_loop.day - 12
    game_hour_sim_loop = current_dt_loop.hour
    game_minute_sim_loop = current_dt_loop.minute

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            save_world_objects() # Save on quit

    advance_game_time(minutes=1)

    # Check for new day
    if game_hour == 0 and game_minute == 0 and current_dt_loop.day != (current_dt_loop - timedelta(minutes=1)).day :
        print(f"\n--- Starting Day {game_day} ---")
        show_message_box(f"--- Starting Day {game_day} ---", BLACK)
        for agent in agents:
            agent.reflect()
            agent.plan_daily_activities()
            agent.update_cached_summary() 

    if game_day > GAME_DAYS_TO_RUN:
        print(f"Simulation finished after {GAME_DAYS_TO_RUN} days.")
        show_message_box(f"Simulation Finished after {GAME_DAYS_TO_RUN} days!", BLACK)
        running = False
        save_world_objects() # Save at end
        break

    for agent in agents:
        agent.update(agents)

    # --- Drawing ---
    SCREEN.fill(WHITE)
    for name, data in LOCATIONS.items():
        if 'Town_' in name or name == 'World': pygame.draw.rect(SCREEN, data['color'], data['rect'])
    for name, data in LOCATIONS.items():
        if 'Town_' not in name and name != 'World':
            pygame.draw.rect(SCREEN, data['color'], data['rect'], border_radius=10)
            pygame.draw.rect(SCREEN, BLACK, data['rect'], 2, border_radius=10) 
            text_surface = FONT.render(name.replace('_', ' '), True, BLACK)
            SCREEN.blit(text_surface, (data['rect'].centerx - text_surface.get_width() / 2, data['rect'].centery - text_surface.get_height() / 2))
            
            # Draw object names and states within their locations
            obj_y_offset = data['rect'].top + 5
            for obj_id, world_obj in WORLD_OBJECTS.items():
                if world_obj.location_name == name:
                    obj_text = f"{world_obj.name}: {world_obj.current_state}"
                    if world_obj.current_user: obj_text += f" (by {world_obj.current_user.name})"
                    obj_surface = SMALL_FONT.render(obj_text, True, DARK_GREY)
                    SCREEN.blit(obj_surface, (data['rect'].left + 5, obj_y_offset))
                    obj_y_offset += 15
                    if obj_y_offset > data['rect'].bottom - 15: break 

    for agent in agents:
        agent.draw(SCREEN)
    time_text = BIG_FONT.render(f"Day: {game_day} Time: {game_hour:02d}:{game_minute:02d}", True, BLACK) 
    SCREEN.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 20, 20))

    if MESSAGE_BOX_TIMER > 0 and MESSAGE_BOX_MESSAGES:
        MESSAGE_BOX_TIMER -= 1
        y_offset = 0
        for msg, color in reversed(MESSAGE_BOX_MESSAGES):
            msg_surface = FONT.render(msg, True, color)
            msg_rect = msg_surface.get_rect(center=(SCREEN_WIDTH // 2, 50 + y_offset))
            bg_rect = msg_rect.inflate(20, 10) 
            pygame.draw.rect(SCREEN, WHITE, bg_rect, border_radius=5)
            pygame.draw.rect(SCREEN, BLACK, bg_rect, 2, border_radius=5) 
            SCREEN.blit(msg_surface, msg_rect)
            y_offset += msg_surface.get_height() + 15 
        if MESSAGE_BOX_TIMER == 0: MESSAGE_BOX_MESSAGES.clear()

    pygame.display.flip()
    clock.tick(FPS)
    if GAME_SPEED_MULTIPLIER > 0:
        time.sleep( (1.0 / GAME_SPEED_MULTIPLIER) / FPS)
    else: pass

# --- Simulation End ---
pygame.quit()

log_filename = f"simulation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(log_filename, 'w') as f:
    json.dump(SIMULATION_LOG, f, indent=2, default=str)
print(f"\nSimulation log saved to {log_filename}")

print("\n--- Final Agent States (Sample) ---")
for agent in agents:
    print(f"\nAgent: {agent.name} ({agent.role}) - Emotion: {agent.emotional_state}")
    print(f"  Needs: {agent.needs}")
    print(f"  Relationships (Top 2 by friendship):")
    sorted_rels = sorted(agent.relationships.items(), key=lambda item: item[1]['friendship_score'], reverse=True)
    for other_name, rel_data in sorted_rels[:2]:
        print(f"    - {other_name}: Friend {rel_data['friendship_score']:.0f}, Trust {rel_data['trust_score']:.0f}")
    print(f"  Memory Count: {len(agent.memory_stream)}")
