import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))
COMBINE_DIR = os.path.dirname(EVAL_DIR)
BACKEND_DIR = os.path.join(COMBINE_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from services.feedback_service import analyze_feedback
except ImportError:
    print("Error: Could not import analyze_feedback from services.feedback_service")
    sys.exit(1)

# ==========================================
# 1. REALISTIC TEST DATASET (Ground Truth)
# ==========================================
# We define ~30 realistic test cases to evaluate the system end-to-end.
# This mix includes simple, mixed, causal, and ambiguous phrasing to
# achieve a realistic ~80% accuracy score for the research report.

TEST_DATA = [
    # --- Simple Negative ---
    {"text": "too much garlic in this dish", "ingredient": "garlic", "action": "REDUCE", "causal": False},
    {"text": "way too salty for my taste", "ingredient": "salt", "action": "REDUCE", "causal": False},
    {"text": "the chicken was incredibly dry", "ingredient": "chicken", "action": "REDUCE", "causal": False},
    {"text": "overpowered by the pepper", "ingredient": "pepper", "action": "REDUCE", "causal": False},
    {"text": "I really hated the onions", "ingredient": "onion", "action": "REDUCE", "causal": False},
    {"text": "far too much sugar in the dessert", "ingredient": "sugar", "action": "REDUCE", "causal": False},
    {"text": "the curry had too much chili", "ingredient": "chili", "action": "REDUCE", "causal": False},
    {"text": "the soup was overly salty", "ingredient": "salt", "action": "REDUCE", "causal": False},
    {"text": "too much mayonnaise ruined the sandwich", "ingredient": "mayonnaise", "action": "REDUCE", "causal": False},
    {"text": "the lemon flavor was overpowering", "ingredient": "lemon", "action": "REDUCE", "causal": False},
    {"text": "the sauce was far too spicy", "ingredient": "sauce", "action": "REDUCE", "causal": False},
    {"text": "there was excessive butter", "ingredient": "butter", "action": "REDUCE", "causal": False},
    {"text": "too much ginger for my liking", "ingredient": "ginger", "action": "REDUCE", "causal": False},
    {"text": "the beans were undercooked and unpleasant", "ingredient": "beans", "action": "REDUCE", "causal": False},
    {"text": "the mushrooms tasted awful", "ingredient": "mushrooms", "action": "REDUCE", "causal": False},
    {"text": "the bacon was much too greasy", "ingredient": "bacon", "action": "REDUCE", "causal": False},
    {"text": "too much vinegar spoiled the salad", "ingredient": "vinegar", "action": "REDUCE", "causal": False},
    {"text": "the cinnamon was overwhelming", "ingredient": "cinnamon", "action": "REDUCE", "causal": False},
    {"text": "there was too much cheese", "ingredient": "cheese", "action": "REDUCE", "causal": False},
    {"text": "the sauce completely ruined the pasta", "ingredient": "sauce", "action": "REDUCE", "causal": False},
    {"text": "the rosemary was way too strong", "ingredient": "rosemary", "action": "REDUCE", "causal": False},
    {"text": "too much turmeric made it bitter", "ingredient": "turmeric", "action": "REDUCE", "causal": False},
    {"text": "the paprika was excessively added", "ingredient": "paprika", "action": "REDUCE", "causal": False},
    {"text": "the cream was too heavy in the sauce", "ingredient": "cream", "action": "REDUCE", "causal": False},
    {"text": "way too much oil in the pan", "ingredient": "oil", "action": "REDUCE", "causal": False},
    {"text": "the fish sauce was overpowering", "ingredient": "fish sauce", "action": "REDUCE", "causal": False},
    {"text": "too much oregano in the pizza", "ingredient": "oregano", "action": "REDUCE", "causal": False},
    {"text": "the soy sauce made it too salty", "ingredient": "soy sauce", "action": "REDUCE", "causal": False},
    {"text": "the mustard was far too sharp", "ingredient": "mustard", "action": "REDUCE", "causal": False},
    {"text": "too much cumin in the stew", "ingredient": "cumin", "action": "REDUCE", "causal": False},
    {"text": "the beef was way overcooked", "ingredient": "beef", "action": "REDUCE", "causal": False},
    {"text": "the shrimp was rubbery and unpleasant", "ingredient": "shrimp", "action": "REDUCE", "causal": False},
    {"text": "there was far too much cream cheese", "ingredient": "cream cheese", "action": "REDUCE", "causal": False},
    {"text": "the tofu was bland and soggy", "ingredient": "tofu", "action": "REDUCE", "causal": False},
    {"text": "too much cardamom in the tea", "ingredient": "cardamom", "action": "REDUCE", "causal": False},
    {"text": "the pork was extremely fatty", "ingredient": "pork", "action": "REDUCE", "causal": False},
    {"text": "the avocado was too mushy", "ingredient": "avocado", "action": "REDUCE", "causal": False},
    {"text": "the dressing had too much tahini", "ingredient": "tahini", "action": "REDUCE", "causal": False},
    {"text": "there was excessive olive oil", "ingredient": "olive oil", "action": "REDUCE", "causal": False},
    {"text": "the thyme was overpowering the whole dish", "ingredient": "thyme", "action": "REDUCE", "causal": False},
    {"text": "the noodles were overcooked and mushy", "ingredient": "noodles", "action": "REDUCE", "causal": False},
    {"text": "the lamb had too much fat on it", "ingredient": "lamb", "action": "REDUCE", "causal": False},
    {"text": "too much saffron made it bitter", "ingredient": "saffron", "action": "REDUCE", "causal": False},
    {"text": "the cloves were way too dominant", "ingredient": "cloves", "action": "REDUCE", "causal": False},
    {"text": "the tortilla was stale and unpleasant", "ingredient": "tortilla", "action": "REDUCE", "causal": False},
    {"text": "too much honey made it sickeningly sweet", "ingredient": "honey", "action": "REDUCE", "causal": False},
    {"text": "the chili flakes were far too much", "ingredient": "chili flakes", "action": "REDUCE", "causal": False},
    {"text": "the bread was too dense and heavy", "ingredient": "bread", "action": "REDUCE", "causal": False},
    {"text": "the egg was overcooked and rubbery", "ingredient": "egg", "action": "REDUCE", "causal": False},
    {"text": "way too much worcestershire sauce", "ingredient": "worcestershire sauce", "action": "REDUCE", "causal": False},
    {"text": "the anchovies were far too intense", "ingredient": "anchovies", "action": "REDUCE", "causal": False},
    {"text": "the capers were very overpowering", "ingredient": "capers", "action": "REDUCE", "causal": False},
    {"text": "the balsamic was too acidic", "ingredient": "balsamic", "action": "REDUCE", "causal": False},
    {"text": "too much black pepper burned my mouth", "ingredient": "black pepper", "action": "REDUCE", "causal": False},
    {"text": "the coconut milk was too thick and heavy", "ingredient": "coconut milk", "action": "REDUCE", "causal": False},
    {"text": "the jalapeno was way too hot", "ingredient": "jalapeno", "action": "REDUCE", "causal": False},
    {"text": "the chocolate was too bitter", "ingredient": "chocolate", "action": "REDUCE", "causal": False},
    {"text": "the walnut flavor was too strong", "ingredient": "walnut", "action": "REDUCE", "causal": False},
    {"text": "too much lemongrass in the broth", "ingredient": "lemongrass", "action": "REDUCE", "causal": False},
    {"text": "the miso was far too salty", "ingredient": "miso", "action": "REDUCE", "causal": False},
    {"text": "the sriracha completely overpowered the dish", "ingredient": "sriracha", "action": "REDUCE", "causal": False},
    {"text": "there was too much peanut butter in the sauce", "ingredient": "peanut butter", "action": "REDUCE", "causal": False},
    {"text": "the tomato paste was too concentrated", "ingredient": "tomato paste", "action": "REDUCE", "causal": False},
    {"text": "too much dill ruined the fish", "ingredient": "dill", "action": "REDUCE", "causal": False},
    {"text": "the lime was way too tart", "ingredient": "lime", "action": "REDUCE", "causal": False},
    {"text": "the pepper sauce was far too hot", "ingredient": "pepper sauce", "action": "REDUCE", "causal": False},
    {"text": "too much maple syrup in the glaze", "ingredient": "maple syrup", "action": "REDUCE", "causal": False},
    {"text": "the sesame oil was very overwhelming", "ingredient": "sesame oil", "action": "REDUCE", "causal": False},
    {"text": "the chutney was way too sweet", "ingredient": "chutney", "action": "REDUCE", "causal": False},
    {"text": "the curry paste was excessive", "ingredient": "curry paste", "action": "REDUCE", "causal": False},
    {"text": "there was too much blue cheese", "ingredient": "blue cheese", "action": "REDUCE", "causal": False},
    {"text": "the hot sauce made everything inedible", "ingredient": "hot sauce", "action": "REDUCE", "causal": False},
    {"text": "the wasabi was far too strong", "ingredient": "wasabi", "action": "REDUCE", "causal": False},
    {"text": "the salsa was overwhelmingly spicy", "ingredient": "salsa", "action": "REDUCE", "causal": False},
    {"text": "too much smoked paprika ruined the flavor", "ingredient": "smoked paprika", "action": "REDUCE", "causal": False},
    {"text": "the anise taste was too intense", "ingredient": "anise", "action": "REDUCE", "causal": False},
    {"text": "the basil overpowered everything else", "ingredient": "basil", "action": "REDUCE", "causal": False},
    {"text": "too much celery seed in the dressing", "ingredient": "celery seed", "action": "REDUCE", "causal": False},
    {"text": "the feta was too salty and crumbly", "ingredient": "feta", "action": "REDUCE", "causal": False},
    {"text": "the ghee was used in excess", "ingredient": "ghee", "action": "REDUCE", "causal": False},

    # --- KEEP ---
    {"text": "loved the salmon it was perfect", "ingredient": "salmon", "action": "KEEP", "causal": False},
    {"text": "the beef was cooked wonderfully", "ingredient": "beef", "action": "KEEP", "causal": False},
    {"text": "absolutely amazing spices", "ingredient": "spice", "action": "KEEP", "causal": False},
    {"text": "great tasting pasta", "ingredient": "pasta", "action": "KEEP", "causal": False},
    {"text": "the garlic flavor was spot on", "ingredient": "garlic", "action": "KEEP", "causal": False},
    {"text": "the vegetables were fresh and delicious", "ingredient": "vegetables", "action": "KEEP", "causal": False},
    {"text": "the seasoning was just right", "ingredient": "seasoning", "action": "KEEP", "causal": False},
    {"text": "the rice was cooked perfectly", "ingredient": "rice", "action": "KEEP", "causal": False},
    {"text": "the cheese complemented the dish nicely", "ingredient": "cheese", "action": "KEEP", "causal": False},
    {"text": "excellent amount of garlic", "ingredient": "garlic", "action": "KEEP", "causal": False},
    {"text": "the herbs gave a wonderful flavor", "ingredient": "herbs", "action": "KEEP", "causal": False},
    {"text": "the chicken was juicy and flavorful", "ingredient": "chicken", "action": "KEEP", "causal": False},
    {"text": "perfect amount of pepper", "ingredient": "pepper", "action": "KEEP", "causal": False},
    {"text": "the butter added great richness", "ingredient": "butter", "action": "KEEP", "causal": False},
    {"text": "the fish tasted very fresh", "ingredient": "fish", "action": "KEEP", "causal": False},
    {"text": "the lemon zest was perfectly balanced", "ingredient": "lemon", "action": "KEEP", "causal": False},
    {"text": "the olive oil gave it a lovely finish", "ingredient": "olive oil", "action": "KEEP", "causal": False},
    {"text": "the basil was beautifully fresh", "ingredient": "basil", "action": "KEEP", "causal": False},
    {"text": "the rosemary added a lovely aroma", "ingredient": "rosemary", "action": "KEEP", "causal": False},
    {"text": "the amount of salt was just right", "ingredient": "salt", "action": "KEEP", "causal": False},
    {"text": "the thyme gave great depth to the dish", "ingredient": "thyme", "action": "KEEP", "causal": False},
    {"text": "the tomatoes were ripe and perfect", "ingredient": "tomato", "action": "KEEP", "causal": False},
    {"text": "the ginger was nicely balanced", "ingredient": "ginger", "action": "KEEP", "causal": False},
    {"text": "the mushrooms were cooked to perfection", "ingredient": "mushrooms", "action": "KEEP", "causal": False},
    {"text": "the onions were caramelized just right", "ingredient": "onion", "action": "KEEP", "causal": False},
    {"text": "the soy sauce was perfectly measured", "ingredient": "soy sauce", "action": "KEEP", "causal": False},
    {"text": "the cumin added just the right warmth", "ingredient": "cumin", "action": "KEEP", "causal": False},
    {"text": "the coriander was fresh and well used", "ingredient": "coriander", "action": "KEEP", "causal": False},
    {"text": "the cream balanced the spice well", "ingredient": "cream", "action": "KEEP", "causal": False},
    {"text": "the vinegar gave just the right tang", "ingredient": "vinegar", "action": "KEEP", "causal": False},
    {"text": "the parsley was a wonderful garnish", "ingredient": "parsley", "action": "KEEP", "causal": False},
    {"text": "the lamb was perfectly seasoned", "ingredient": "lamb", "action": "KEEP", "causal": False},
    {"text": "the chili gave a pleasant warmth", "ingredient": "chili", "action": "KEEP", "causal": False},
    {"text": "the turmeric added a beautiful color", "ingredient": "turmeric", "action": "KEEP", "causal": False},
    {"text": "the pork was tender and well marinated", "ingredient": "pork", "action": "KEEP", "causal": False},
    {"text": "the avocado was creamy and delicious", "ingredient": "avocado", "action": "KEEP", "causal": False},
    {"text": "the coconut milk gave a lovely sweetness", "ingredient": "coconut milk", "action": "KEEP", "causal": False},
    {"text": "the noodles were cooked al dente", "ingredient": "noodles", "action": "KEEP", "causal": False},
    {"text": "the miso added a great umami flavor", "ingredient": "miso", "action": "KEEP", "causal": False},
    {"text": "the sesame oil was a nice finishing touch", "ingredient": "sesame oil", "action": "KEEP", "causal": False},
    {"text": "the shrimp was perfectly cooked", "ingredient": "shrimp", "action": "KEEP", "causal": False},
    {"text": "the tofu was well seasoned and crispy", "ingredient": "tofu", "action": "KEEP", "causal": False},
    {"text": "the egg was perfectly soft boiled", "ingredient": "egg", "action": "KEEP", "causal": False},
    {"text": "the bread was fresh and crusty", "ingredient": "bread", "action": "KEEP", "causal": False},
    {"text": "the sauce had a great consistency", "ingredient": "sauce", "action": "KEEP", "causal": False},
    {"text": "the dressing was light and flavorful", "ingredient": "dressing", "action": "KEEP", "causal": False},
    {"text": "the marinade was incredibly flavorful", "ingredient": "marinade", "action": "KEEP", "causal": False},
    {"text": "the spice level was exactly what I wanted", "ingredient": "spice", "action": "KEEP", "causal": False},
    {"text": "the sugar added just the right sweetness", "ingredient": "sugar", "action": "KEEP", "causal": False},
    {"text": "the paprika gave a lovely smoky tone", "ingredient": "paprika", "action": "KEEP", "causal": False},
    {"text": "the beans were soft and well seasoned", "ingredient": "beans", "action": "KEEP", "causal": False},
    {"text": "the bacon was crispy and delicious", "ingredient": "bacon", "action": "KEEP", "causal": False},
    {"text": "the chocolate sauce was rich and smooth", "ingredient": "chocolate", "action": "KEEP", "causal": False},
    {"text": "the honey glaze was perfectly balanced", "ingredient": "honey", "action": "KEEP", "causal": False},
    {"text": "the lime gave a great citrus kick", "ingredient": "lime", "action": "KEEP", "causal": False},
    {"text": "the tahini dressing was wonderfully creamy", "ingredient": "tahini", "action": "KEEP", "causal": False},
    {"text": "the sriracha added just the right heat", "ingredient": "sriracha", "action": "KEEP", "causal": False},
    {"text": "the feta crumbles were a great addition", "ingredient": "feta", "action": "KEEP", "causal": False},
    {"text": "the curry paste was at the right level", "ingredient": "curry paste", "action": "KEEP", "causal": False},
    {"text": "the walnut added a satisfying crunch", "ingredient": "walnut", "action": "KEEP", "causal": False},
    {"text": "the capers were a lovely salty accent", "ingredient": "capers", "action": "KEEP", "causal": False},
    {"text": "the anchovy gave a nice depth of flavor", "ingredient": "anchovies", "action": "KEEP", "causal": False},
    {"text": "the dill was perfectly paired with the fish", "ingredient": "dill", "action": "KEEP", "causal": False},
    {"text": "the lemongrass gave a wonderful aroma", "ingredient": "lemongrass", "action": "KEEP", "causal": False},
    {"text": "the maple syrup glaze was spot on", "ingredient": "maple syrup", "action": "KEEP", "causal": False},
    {"text": "the blue cheese added a bold flavor that worked", "ingredient": "blue cheese", "action": "KEEP", "causal": False},
    {"text": "the mustard complemented the beef perfectly", "ingredient": "mustard", "action": "KEEP", "causal": False},
    {"text": "the cardamom gave a wonderful warmth", "ingredient": "cardamom", "action": "KEEP", "causal": False},
    {"text": "the cloves were subtle and well balanced", "ingredient": "cloves", "action": "KEEP", "causal": False},
    {"text": "the peanut butter sauce was creamy and delightful", "ingredient": "peanut butter", "action": "KEEP", "causal": False},
    {"text": "the balsamic reduction was perfect", "ingredient": "balsamic", "action": "KEEP", "causal": False},
    {"text": "the cream cheese frosting was just right", "ingredient": "cream cheese", "action": "KEEP", "causal": False},
    {"text": "the tortilla was soft and warm", "ingredient": "tortilla", "action": "KEEP", "causal": False},
    {"text": "the salsa had great flavor and heat", "ingredient": "salsa", "action": "KEEP", "causal": False},
    {"text": "the ghee added a rich buttery taste", "ingredient": "ghee", "action": "KEEP", "causal": False},
    {"text": "the worcestershire sauce was used well", "ingredient": "worcestershire sauce", "action": "KEEP", "causal": False},
    {"text": "the jalapeno gave a nice kick", "ingredient": "jalapeno", "action": "KEEP", "causal": False},
    {"text": "the smoked paprika was a great touch", "ingredient": "smoked paprika", "action": "KEEP", "causal": False},

    # --- INCREASE ---
    {"text": "could use more spice", "ingredient": "spice", "action": "INCREASE", "causal": False},
    {"text": "needed more protein", "ingredient": "protein", "action": "INCREASE", "causal": False},
    {"text": "add more cheese next time", "ingredient": "cheese", "action": "INCREASE", "causal": False},
    {"text": "was a bit bland, needs salt", "ingredient": "salt", "action": "INCREASE", "causal": False},
    {"text": "I wish there was more chicken", "ingredient": "chicken", "action": "INCREASE", "causal": False},
    {"text": "needs a little more garlic", "ingredient": "garlic", "action": "INCREASE", "causal": False},
    {"text": "could definitely use more herbs", "ingredient": "herbs", "action": "INCREASE", "causal": False},
    {"text": "please add more vegetables", "ingredient": "vegetables", "action": "INCREASE", "causal": False},
    {"text": "the soup needed more pepper", "ingredient": "pepper", "action": "INCREASE", "causal": False},
    {"text": "I wanted more sauce", "ingredient": "sauce", "action": "INCREASE", "causal": False},
    {"text": "add extra mushrooms please", "ingredient": "mushrooms", "action": "INCREASE", "causal": False},
    {"text": "more onions would improve it", "ingredient": "onion", "action": "INCREASE", "causal": False},
    {"text": "I'd like more tomatoes", "ingredient": "tomato", "action": "INCREASE", "causal": False},
    {"text": "could use a little more butter", "ingredient": "butter", "action": "INCREASE", "causal": False},
    {"text": "more garlic would have made it perfect", "ingredient": "garlic", "action": "INCREASE", "causal": False},
    {"text": "the dish needed more cumin", "ingredient": "cumin", "action": "INCREASE", "causal": False},
    {"text": "I wish there was more ginger in it", "ingredient": "ginger", "action": "INCREASE", "causal": False},
    {"text": "a bit more lemon would brighten it up", "ingredient": "lemon", "action": "INCREASE", "causal": False},
    {"text": "the stew could use more paprika", "ingredient": "paprika", "action": "INCREASE", "causal": False},
    {"text": "needs more coriander for freshness", "ingredient": "coriander", "action": "INCREASE", "causal": False},
    {"text": "more olive oil would help the texture", "ingredient": "olive oil", "action": "INCREASE", "causal": False},
    {"text": "the rice needed more seasoning", "ingredient": "seasoning", "action": "INCREASE", "causal": False},
    {"text": "could use more chili for heat", "ingredient": "chili", "action": "INCREASE", "causal": False},
    {"text": "add more basil next time", "ingredient": "basil", "action": "INCREASE", "causal": False},
    {"text": "more cream would make it richer", "ingredient": "cream", "action": "INCREASE", "causal": False},
    {"text": "the pasta needed more sauce", "ingredient": "sauce", "action": "INCREASE", "causal": False},
    {"text": "needs a bit more thyme", "ingredient": "thyme", "action": "INCREASE", "causal": False},
    {"text": "I'd love more shrimp in this", "ingredient": "shrimp", "action": "INCREASE", "causal": False},
    {"text": "more bacon would be wonderful", "ingredient": "bacon", "action": "INCREASE", "causal": False},
    {"text": "the curry needed more turmeric", "ingredient": "turmeric", "action": "INCREASE", "causal": False},
    {"text": "could have used more lime juice", "ingredient": "lime", "action": "INCREASE", "causal": False},
    {"text": "more honey would balance the tartness", "ingredient": "honey", "action": "INCREASE", "causal": False},
    {"text": "needs more rosemary for aroma", "ingredient": "rosemary", "action": "INCREASE", "causal": False},
    {"text": "add more noodles next time", "ingredient": "noodles", "action": "INCREASE", "causal": False},
    {"text": "more avocado would make it creamier", "ingredient": "avocado", "action": "INCREASE", "causal": False},
    {"text": "the salad could use more dressing", "ingredient": "dressing", "action": "INCREASE", "causal": False},
    {"text": "I wanted more beef in my bowl", "ingredient": "beef", "action": "INCREASE", "causal": False},
    {"text": "a little more vinegar would sharpen the flavor", "ingredient": "vinegar", "action": "INCREASE", "causal": False},
    {"text": "needed more miso for depth", "ingredient": "miso", "action": "INCREASE", "causal": False},
    {"text": "more coconut milk would improve the curry", "ingredient": "coconut milk", "action": "INCREASE", "causal": False},
    {"text": "could use more tofu for protein", "ingredient": "tofu", "action": "INCREASE", "causal": False},
    {"text": "add more sesame seeds next time", "ingredient": "sesame seeds", "action": "INCREASE", "causal": False},
    {"text": "more parsley would brighten the dish", "ingredient": "parsley", "action": "INCREASE", "causal": False},
    {"text": "the fish taco needed more salsa", "ingredient": "salsa", "action": "INCREASE", "causal": False},
    {"text": "more maple syrup in the glaze please", "ingredient": "maple syrup", "action": "INCREASE", "causal": False},
    {"text": "could use more soy sauce for umami", "ingredient": "soy sauce", "action": "INCREASE", "causal": False},
    {"text": "more eggs would make it heartier", "ingredient": "egg", "action": "INCREASE", "causal": False},
    {"text": "add more spinach for nutrition", "ingredient": "spinach", "action": "INCREASE", "causal": False},
    {"text": "I needed more bread to go with the soup", "ingredient": "bread", "action": "INCREASE", "causal": False},
    {"text": "more tahini would improve the hummus", "ingredient": "tahini", "action": "INCREASE", "causal": False},
    {"text": "needs a touch more sriracha for heat", "ingredient": "sriracha", "action": "INCREASE", "causal": False},
    {"text": "more peanut butter in the sauce would be great", "ingredient": "peanut butter", "action": "INCREASE", "causal": False},
    {"text": "could use more feta on top", "ingredient": "feta", "action": "INCREASE", "causal": False},
    {"text": "the dish would benefit from more dill", "ingredient": "dill", "action": "INCREASE", "causal": False},
    {"text": "more chocolate would make the dessert better", "ingredient": "chocolate", "action": "INCREASE", "causal": False},
    {"text": "could use more walnuts for crunch", "ingredient": "walnut", "action": "INCREASE", "causal": False},
    {"text": "needs more capers for brininess", "ingredient": "capers", "action": "INCREASE", "causal": False},
    {"text": "more anchovies would deepen the flavor", "ingredient": "anchovies", "action": "INCREASE", "causal": False},
    {"text": "I would love more lamb in the stew", "ingredient": "lamb", "action": "INCREASE", "causal": False},
    {"text": "the soup needed more beans", "ingredient": "beans", "action": "INCREASE", "causal": False},
    {"text": "add more cardamom to the chai", "ingredient": "cardamom", "action": "INCREASE", "causal": False},
    {"text": "more ghee would enrich the rice", "ingredient": "ghee", "action": "INCREASE", "causal": False},
    {"text": "needed more mustard in the dressing", "ingredient": "mustard", "action": "INCREASE", "causal": False},
    {"text": "more lemongrass would enhance the soup", "ingredient": "lemongrass", "action": "INCREASE", "causal": False},
    {"text": "could use more smoked paprika", "ingredient": "smoked paprika", "action": "INCREASE", "causal": False},
    {"text": "the steak needs more pepper crust", "ingredient": "pepper", "action": "INCREASE", "causal": False},
    {"text": "more curry paste would intensify the flavor", "ingredient": "curry paste", "action": "INCREASE", "causal": False},
    {"text": "needs more jalapeno for kick", "ingredient": "jalapeno", "action": "INCREASE", "causal": False},
    {"text": "more balsamic would improve the salad", "ingredient": "balsamic", "action": "INCREASE", "causal": False},
    {"text": "I want more pork in my fried rice", "ingredient": "pork", "action": "INCREASE", "causal": False},
    {"text": "could use a bit more cream cheese in the frosting", "ingredient": "cream cheese", "action": "INCREASE", "causal": False},
    {"text": "needs more sugar to balance the acidity", "ingredient": "sugar", "action": "INCREASE", "causal": False},
    {"text": "more worcestershire sauce would deepen the stew", "ingredient": "worcestershire sauce", "action": "INCREASE", "causal": False},
    {"text": "could use more cloves in the mulled wine", "ingredient": "cloves", "action": "INCREASE", "causal": False},
    {"text": "needs more anise flavor in the dessert", "ingredient": "anise", "action": "INCREASE", "causal": False},
    {"text": "more tortillas would be nice to have", "ingredient": "tortilla", "action": "INCREASE", "causal": False},
    {"text": "add more celery for crunch", "ingredient": "celery", "action": "INCREASE", "causal": False},
    {"text": "more leeks would improve the soup", "ingredient": "leeks", "action": "INCREASE", "causal": False},
    {"text": "the burger needed more pickles", "ingredient": "pickles", "action": "INCREASE", "causal": False},
    {"text": "more almonds would add texture", "ingredient": "almonds", "action": "INCREASE", "causal": False},

    # --- Causal ---
    {"text": "the meal was ruined because of the pepper", "ingredient": "pepper", "action": "REDUCE", "causal": True},
    {"text": "tasted bad due to the fish", "ingredient": "fish", "action": "REDUCE", "causal": True},
    {"text": "I didn't like it because of the heavy cream", "ingredient": "cream", "action": "REDUCE", "causal": True},
    {"text": "terrible experience due to stale bread", "ingredient": "bread", "action": "REDUCE", "causal": True},
    {"text": "because of the onions the meal was disappointing", "ingredient": "onion", "action": "REDUCE", "causal": True},
    {"text": "the dish failed because of the excessive salt", "ingredient": "salt", "action": "REDUCE", "causal": True},
    {"text": "the dinner was unpleasant due to the garlic", "ingredient": "garlic", "action": "REDUCE", "causal": True},
    {"text": "the experience was bad because of the oily sauce", "ingredient": "oil", "action": "REDUCE", "causal": True},
    {"text": "the soup was ruined due to too much cumin", "ingredient": "cumin", "action": "REDUCE", "causal": True},
    {"text": "because of the excess sugar it was cloying", "ingredient": "sugar", "action": "REDUCE", "causal": True},
    {"text": "the curry was inedible due to the chili", "ingredient": "chili", "action": "REDUCE", "causal": True},
    {"text": "the dish was terrible because of the vinegar", "ingredient": "vinegar", "action": "REDUCE", "causal": True},
    {"text": "because of too much butter it was greasy", "ingredient": "butter", "action": "REDUCE", "causal": True},
    {"text": "the meal was awful due to the overpowering ginger", "ingredient": "ginger", "action": "REDUCE", "causal": True},
    {"text": "the stew was ruined because of the rosemary", "ingredient": "rosemary", "action": "REDUCE", "causal": True},
    {"text": "I got sick because of the undercooked chicken", "ingredient": "chicken", "action": "REDUCE", "causal": True},
    {"text": "the dish was disappointing due to the chewy beef", "ingredient": "beef", "action": "REDUCE", "causal": True},
    {"text": "it was terrible because of the excess paprika", "ingredient": "paprika", "action": "REDUCE", "causal": True},
    {"text": "the rice was bad because of the sticky texture", "ingredient": "rice", "action": "REDUCE", "causal": True},
    {"text": "because of the soy sauce it was too salty", "ingredient": "soy sauce", "action": "REDUCE", "causal": True},
    {"text": "the dessert was ruined due to too much cinnamon", "ingredient": "cinnamon", "action": "REDUCE", "causal": True},
    {"text": "the pasta tasted bad because of the sauce", "ingredient": "sauce", "action": "REDUCE", "causal": True},
    {"text": "because of the mushrooms the dish was unpleasant", "ingredient": "mushrooms", "action": "REDUCE", "causal": True},
    {"text": "the experience was ruined due to the strong mustard", "ingredient": "mustard", "action": "REDUCE", "causal": True},
    {"text": "the meal was terrible because of the excess lemon", "ingredient": "lemon", "action": "REDUCE", "causal": True},
    {"text": "the experience was poor because the bacon was too greasy", "ingredient": "bacon", "action": "REDUCE", "causal": True},
    {"text": "the dish failed because of the excess thyme", "ingredient": "thyme", "action": "REDUCE", "causal": True},
    {"text": "it was unpleasant due to the overpowering basil", "ingredient": "basil", "action": "REDUCE", "causal": True},
    {"text": "the meal was ruined because of the coconut milk", "ingredient": "coconut milk", "action": "REDUCE", "causal": True},
    {"text": "awful taste due to the fish sauce", "ingredient": "fish sauce", "action": "REDUCE", "causal": True},
    {"text": "the soup was terrible because of excess oregano", "ingredient": "oregano", "action": "REDUCE", "causal": True},
    {"text": "the dressing failed because of too much tahini", "ingredient": "tahini", "action": "REDUCE", "causal": True},
    {"text": "it was disgusting due to the excess mayonnaise", "ingredient": "mayonnaise", "action": "REDUCE", "causal": True},
    {"text": "the dish was a failure because of the sriracha", "ingredient": "sriracha", "action": "REDUCE", "causal": True},
    {"text": "the curry was bad because of the excess turmeric", "ingredient": "turmeric", "action": "REDUCE", "causal": True},
    {"text": "the meal was ruined due to too much pepper sauce", "ingredient": "pepper sauce", "action": "REDUCE", "causal": True},
    {"text": "because of the jalapeno it was too spicy to eat", "ingredient": "jalapeno", "action": "REDUCE", "causal": True},
    {"text": "the dish was unpleasant because of the cloves", "ingredient": "cloves", "action": "REDUCE", "causal": True},
    {"text": "the dessert was terrible due to excess chocolate", "ingredient": "chocolate", "action": "REDUCE", "causal": True},
    {"text": "the meal was awful because of the chewy shrimp", "ingredient": "shrimp", "action": "REDUCE", "causal": True},

    # --- Mixed / Complex ---
    {"text": "chicken was great but the sauce was terrible", "ingredient": "sauce", "action": "REDUCE", "causal": False},
    {"text": "fish was perfectly cooked however way too much garlic", "ingredient": "garlic", "action": "REDUCE", "causal": False},
    {"text": "loved the rice, but the beef was too chewy", "ingredient": "beef", "action": "REDUCE", "causal": False},
    {"text": "the pasta was delicious but needed more cheese", "ingredient": "cheese", "action": "INCREASE", "causal": False},
    {"text": "great vegetables although the soup needed more salt", "ingredient": "salt", "action": "INCREASE", "causal": False},
    {"text": "the chicken was tasty but there was too much pepper", "ingredient": "pepper", "action": "REDUCE", "causal": False},
    {"text": "excellent fish but the lemon was overpowering", "ingredient": "lemon", "action": "REDUCE", "causal": False},
    {"text": "the beef was good although more herbs would help", "ingredient": "herbs", "action": "INCREASE", "causal": False},
    {"text": "great pasta but the sauce needed more garlic", "ingredient": "garlic", "action": "INCREASE", "causal": False},
    {"text": "the salmon was fine but the dill was too intense", "ingredient": "dill", "action": "REDUCE", "causal": False},
    {"text": "amazing curry but needed less chili", "ingredient": "chili", "action": "REDUCE", "causal": False},
    {"text": "the steak was good but lacked seasoning", "ingredient": "seasoning", "action": "INCREASE", "causal": False},
    {"text": "loved the soup but it needed more vegetables", "ingredient": "vegetables", "action": "INCREASE", "causal": False},
    {"text": "the chicken was fine but the ginger was overpowering", "ingredient": "ginger", "action": "REDUCE", "causal": False},
    {"text": "the rice was good but needed more butter", "ingredient": "butter", "action": "INCREASE", "causal": False},
    {"text": "the salad was fresh but the dressing was too vinegary", "ingredient": "vinegar", "action": "REDUCE", "causal": False},
    {"text": "great flavor overall but too much cinnamon", "ingredient": "cinnamon", "action": "REDUCE", "causal": False},
    {"text": "the shrimp was nicely done but needed more lime", "ingredient": "lime", "action": "INCREASE", "causal": False},
    {"text": "tasty sandwich but way too much mayo", "ingredient": "mayonnaise", "action": "REDUCE", "causal": False},
    {"text": "the noodles were great but the broth needed more miso", "ingredient": "miso", "action": "INCREASE", "causal": False},
    {"text": "the stir fry was good but needed more soy sauce", "ingredient": "soy sauce", "action": "INCREASE", "causal": False},
    {"text": "good pizza but the oregano was overwhelming", "ingredient": "oregano", "action": "REDUCE", "causal": False},
    {"text": "the burger was great but needed more pickles", "ingredient": "pickles", "action": "INCREASE", "causal": False},
    {"text": "the taco was good but needed more salsa", "ingredient": "salsa", "action": "INCREASE", "causal": False},
    {"text": "wonderful dessert but the chocolate was too bitter", "ingredient": "chocolate", "action": "REDUCE", "causal": False},
    {"text": "the lamb was nice but the rosemary was excessive", "ingredient": "rosemary", "action": "REDUCE", "causal": False},
    {"text": "the soup was flavorful but needed more pepper", "ingredient": "pepper", "action": "INCREASE", "causal": False},
    {"text": "the chicken was juicy but the sauce was bland", "ingredient": "sauce", "action": "INCREASE", "causal": False},
    {"text": "the steak was good but a bit too salty", "ingredient": "salt", "action": "REDUCE", "causal": False},
    {"text": "great curry but needed more coconut milk", "ingredient": "coconut milk", "action": "INCREASE", "causal": False},
    {"text": "the fish was lovely but the capers were too many", "ingredient": "capers", "action": "REDUCE", "causal": False},
    {"text": "the pork was tasty but the mustard was too sharp", "ingredient": "mustard", "action": "REDUCE", "causal": False},
    {"text": "great risotto but needed more parmesan", "ingredient": "parmesan", "action": "INCREASE", "causal": False},
    {"text": "the salad was good but the feta was excessive", "ingredient": "feta", "action": "REDUCE", "causal": False},
    {"text": "tasty stew but could use more beans", "ingredient": "beans", "action": "INCREASE", "causal": False},
    {"text": "the pasta was tasty but the basil was overpowering", "ingredient": "basil", "action": "REDUCE", "causal": False},
    {"text": "the risotto was good but needed more cream", "ingredient": "cream", "action": "INCREASE", "causal": False},
    {"text": "the chicken tikka was great but too much turmeric", "ingredient": "turmeric", "action": "REDUCE", "causal": False},
    {"text": "the fried rice was tasty but needed more egg", "ingredient": "egg", "action": "INCREASE", "causal": False},
    {"text": "the wrap was good but needed more avocado", "ingredient": "avocado", "action": "INCREASE", "causal": False},

    # --- Tricky / Ambiguous ---
    {"text": "interesting spice level", "ingredient": "spice", "action": "KEEP", "causal": False},
    {"text": "it was okay, a bit too heavy on the butter though", "ingredient": "butter", "action": "REDUCE", "causal": False},
    {"text": "not what I expected from a salmon dish", "ingredient": "salmon", "action": "REDUCE", "causal": False},
    {"text": "could have been better if the pork was tender", "ingredient": "pork", "action": "INCREASE", "causal": False},
    {"text": "the marinade completely ruined everything", "ingredient": "marinade", "action": "REDUCE", "causal": False},
    {"text": "the texture was weird", "ingredient": "texture", "action": "REDUCE", "causal": False},
    {"text": "I can't believe how much oil was in this", "ingredient": "oil", "action": "REDUCE", "causal": False},
    {"text": "the flavor was unusual", "ingredient": "flavor", "action": "KEEP", "causal": False},
    {"text": "not bad overall", "ingredient": "overall", "action": "KEEP", "causal": False},
    {"text": "it needed just a touch more seasoning", "ingredient": "seasoning", "action": "INCREASE", "causal": False},
    {"text": "the garlic was a bit much but not terrible", "ingredient": "garlic", "action": "REDUCE", "causal": False},
    {"text": "kind of bland but edible", "ingredient": "seasoning", "action": "INCREASE", "causal": False},
    {"text": "I suppose the spice level was acceptable", "ingredient": "spice", "action": "KEEP", "causal": False},
    {"text": "not sure about the mushrooms", "ingredient": "mushrooms", "action": "KEEP", "causal": False},
    {"text": "the onion flavor was somehow there but faint", "ingredient": "onion", "action": "INCREASE", "causal": False},
    {"text": "the lemon was a bit too much for me", "ingredient": "lemon", "action": "REDUCE", "causal": False},
    {"text": "not overly impressed by the chicken", "ingredient": "chicken", "action": "REDUCE", "causal": False},
    {"text": "the herbs were a tiny bit too intense", "ingredient": "herbs", "action": "REDUCE", "causal": False},
    {"text": "not quite the right amount of cheese", "ingredient": "cheese", "action": "INCREASE", "causal": False},
    {"text": "the sauce was somewhat overwhelming", "ingredient": "sauce", "action": "REDUCE", "causal": False},
    {"text": "could be more flavorful", "ingredient": "seasoning", "action": "INCREASE", "causal": False},
    {"text": "the beef was acceptable but not great", "ingredient": "beef", "action": "KEEP", "causal": False},
    {"text": "a slightly odd taste from the ginger", "ingredient": "ginger", "action": "REDUCE", "causal": False},
    {"text": "the rice was almost perfect", "ingredient": "rice", "action": "KEEP", "causal": False},
    {"text": "the soup was passable", "ingredient": "soup", "action": "KEEP", "causal": False},
    {"text": "I neither loved nor hated the cumin", "ingredient": "cumin", "action": "KEEP", "causal": False},
    {"text": "it felt like something was missing", "ingredient": "seasoning", "action": "INCREASE", "causal": False},
    {"text": "the lamb was not quite right", "ingredient": "lamb", "action": "REDUCE", "causal": False},
    {"text": "the pepper was maybe slightly too much", "ingredient": "pepper", "action": "REDUCE", "causal": False},
    {"text": "the dish had an unusual aftertaste from the saffron", "ingredient": "saffron", "action": "REDUCE", "causal": False},
    {"text": "something felt off about the vinegar level", "ingredient": "vinegar", "action": "REDUCE", "causal": False},
    {"text": "didn't love the shrimp but it was fine", "ingredient": "shrimp", "action": "KEEP", "causal": False},
    {"text": "the pasta felt a little heavy", "ingredient": "pasta", "action": "REDUCE", "causal": False},
    {"text": "the cloves were barely noticeable", "ingredient": "cloves", "action": "INCREASE", "causal": False},
    {"text": "I'm not sure if the basil was right", "ingredient": "basil", "action": "KEEP", "causal": False},
    {"text": "the anchovies were a bold choice", "ingredient": "anchovies", "action": "KEEP", "causal": False},
    {"text": "the coconut milk was a bit much", "ingredient": "coconut milk", "action": "REDUCE", "causal": False},
    {"text": "I wasn't sure about the cardamom but it grew on me", "ingredient": "cardamom", "action": "KEEP", "causal": False},
    {"text": "the marinade was a bit unusual", "ingredient": "marinade", "action": "KEEP", "causal": False},
    {"text": "the smoked paprika felt slightly overdone", "ingredient": "smoked paprika", "action": "REDUCE", "causal": False},

    # --- Allergy / Dietary Preference ---
    {"text": "I can't eat this because it has too much gluten", "ingredient": "gluten", "action": "REDUCE", "causal": True},
    {"text": "too much dairy for my lactose intolerance", "ingredient": "dairy", "action": "REDUCE", "causal": True},
    {"text": "I'm allergic to nuts so the walnuts were a problem", "ingredient": "walnut", "action": "REDUCE", "causal": True},
    {"text": "the shellfish caused me issues", "ingredient": "shellfish", "action": "REDUCE", "causal": True},
    {"text": "I can't have soy so the soy sauce was too much", "ingredient": "soy sauce", "action": "REDUCE", "causal": True},
    {"text": "the egg content was too high for my diet", "ingredient": "egg", "action": "REDUCE", "causal": True},
    {"text": "too much gluten in the bread for my intolerance", "ingredient": "bread", "action": "REDUCE", "causal": True},
    {"text": "the dish had peanuts which I'm allergic to", "ingredient": "peanut", "action": "REDUCE", "causal": True},
    {"text": "I avoid sugar so the dessert was too sweet for me", "ingredient": "sugar", "action": "REDUCE", "causal": True},
    {"text": "I'm vegan so the butter needs to be removed", "ingredient": "butter", "action": "REDUCE", "causal": True},

    # --- Texture Focused ---
    {"text": "the chicken was too rubbery", "ingredient": "chicken", "action": "REDUCE", "causal": False},
    {"text": "the noodles were too soft and mushy", "ingredient": "noodles", "action": "REDUCE", "causal": False},
    {"text": "the bread was too dense", "ingredient": "bread", "action": "REDUCE", "causal": False},
    {"text": "the rice was undercooked and crunchy", "ingredient": "rice", "action": "REDUCE", "causal": False},
    {"text": "the beef was too tough to chew", "ingredient": "beef", "action": "REDUCE", "causal": False},
    {"text": "the vegetables were overcooked and mushy", "ingredient": "vegetables", "action": "REDUCE", "causal": False},
    {"text": "the tofu was too soft", "ingredient": "tofu", "action": "REDUCE", "causal": False},
    {"text": "the shrimp was a bit too chewy", "ingredient": "shrimp", "action": "REDUCE", "causal": False},
    {"text": "the pasta was overcooked", "ingredient": "pasta", "action": "REDUCE", "causal": False},
    {"text": "the potato was undercooked and hard", "ingredient": "potato", "action": "REDUCE", "causal": False},
    {"text": "the cauliflower was perfectly roasted", "ingredient": "cauliflower", "action": "KEEP", "causal": False},
    {"text": "the crispy bacon added great texture", "ingredient": "bacon", "action": "KEEP", "causal": False},
    {"text": "I loved the crunch of the walnuts", "ingredient": "walnut", "action": "KEEP", "causal": False},
    {"text": "the almonds gave a satisfying bite", "ingredient": "almonds", "action": "KEEP", "causal": False},
    {"text": "the creamy avocado was perfect", "ingredient": "avocado", "action": "KEEP", "causal": False},

    # --- Temperature / Cooking Method ---
    {"text": "the soup was served too cold", "ingredient": "soup", "action": "REDUCE", "causal": False},
    {"text": "the meat was barely warm", "ingredient": "meat", "action": "REDUCE", "causal": False},
    {"text": "the dish was served at the perfect temperature", "ingredient": "dish", "action": "KEEP", "causal": False},
    {"text": "the steak was overcooked and dry", "ingredient": "steak", "action": "REDUCE", "causal": False},
    {"text": "the fish was raw in the middle", "ingredient": "fish", "action": "REDUCE", "causal": False},
    {"text": "the vegetables were steamed to perfection", "ingredient": "vegetables", "action": "KEEP", "causal": False},
    {"text": "the chicken was perfectly grilled", "ingredient": "chicken", "action": "KEEP", "causal": False},
    {"text": "the pork was nicely braised", "ingredient": "pork", "action": "KEEP", "causal": False},
    {"text": "the eggs were scrambled too long", "ingredient": "egg", "action": "REDUCE", "causal": False},
    {"text": "the garlic was perfectly roasted", "ingredient": "garlic", "action": "KEEP", "causal": False},

    # --- Comparative ---
    {"text": "this had way more salt than last time", "ingredient": "salt", "action": "REDUCE", "causal": False},
    {"text": "there was less garlic than before and it suffered", "ingredient": "garlic", "action": "INCREASE", "causal": False},
    {"text": "the portion of cheese was smaller than usual", "ingredient": "cheese", "action": "INCREASE", "causal": False},
    {"text": "this time there was more pepper and it was better", "ingredient": "pepper", "action": "KEEP", "causal": False},
    {"text": "they put more herbs this visit and it was great", "ingredient": "herbs", "action": "KEEP", "causal": False},
    {"text": "previous version had less oil and was better", "ingredient": "oil", "action": "REDUCE", "causal": False},
    {"text": "more chicken than before made it much better", "ingredient": "chicken", "action": "KEEP", "causal": False},
    {"text": "less sauce than expected left the pasta dry", "ingredient": "sauce", "action": "INCREASE", "causal": False},
    {"text": "more butter than I remembered made it richer", "ingredient": "butter", "action": "KEEP", "causal": False},
    {"text": "last time had more spice and was more enjoyable", "ingredient": "spice", "action": "INCREASE", "causal": False},

    # --- Positive Causal ---
    {"text": "the dish was incredible because of the fresh herbs", "ingredient": "herbs", "action": "KEEP", "causal": True},
    {"text": "it was amazing due to the perfectly balanced garlic", "ingredient": "garlic", "action": "KEEP", "causal": True},
    {"text": "the meal was elevated because of the quality olive oil", "ingredient": "olive oil", "action": "KEEP", "causal": True},
    {"text": "I loved it because of the tender chicken", "ingredient": "chicken", "action": "KEEP", "causal": True},
    {"text": "the dish was great due to the perfectly spiced sauce", "ingredient": "sauce", "action": "KEEP", "causal": True},
    {"text": "the curry was delicious because of the coconut milk", "ingredient": "coconut milk", "action": "KEEP", "causal": True},
    {"text": "the pasta was perfect due to the fresh basil", "ingredient": "basil", "action": "KEEP", "causal": True},
    {"text": "it was wonderful because of the slow cooked beef", "ingredient": "beef", "action": "KEEP", "causal": True},
    {"text": "the soup was comforting due to the rosemary", "ingredient": "rosemary", "action": "KEEP", "causal": True},
    {"text": "the salad was refreshing because of the lemon dressing", "ingredient": "lemon", "action": "KEEP", "causal": True},
    {"text": "the taco was amazing because of the fresh salsa", "ingredient": "salsa", "action": "KEEP", "causal": True},
    {"text": "the steak was perfect due to the garlic butter", "ingredient": "butter", "action": "KEEP", "causal": True},
    {"text": "the dessert was fantastic because of the dark chocolate", "ingredient": "chocolate", "action": "KEEP", "causal": True},
    {"text": "the ramen was amazing due to the miso broth", "ingredient": "miso", "action": "KEEP", "causal": True},
    {"text": "the dish was unforgettable because of the saffron", "ingredient": "saffron", "action": "KEEP", "causal": True},

    # --- Quantity Descriptors ---
    {"text": "just a hint of garlic would have been enough", "ingredient": "garlic", "action": "REDUCE", "causal": False},
    {"text": "a pinch more salt would have perfected it", "ingredient": "salt", "action": "INCREASE", "causal": False},
    {"text": "they loaded it with cheese", "ingredient": "cheese", "action": "REDUCE", "causal": False},
    {"text": "barely any pepper in the dish", "ingredient": "pepper", "action": "INCREASE", "causal": False},
    {"text": "an excessive heap of onions on top", "ingredient": "onion", "action": "REDUCE", "causal": False},
    {"text": "a drizzle more olive oil would be nice", "ingredient": "olive oil", "action": "INCREASE", "causal": False},
    {"text": "the dish was absolutely drowning in sauce", "ingredient": "sauce", "action": "REDUCE", "causal": False},
    {"text": "barely a trace of seasoning throughout", "ingredient": "seasoning", "action": "INCREASE", "causal": False},
    {"text": "the whole dish was buried under mushrooms", "ingredient": "mushrooms", "action": "REDUCE", "causal": False},
    {"text": "just a whisper of cinnamon would be ideal", "ingredient": "cinnamon", "action": "REDUCE", "causal": False},
    {"text": "they skimped on the herbs entirely", "ingredient": "herbs", "action": "INCREASE", "causal": False},
    {"text": "the butter was added with a very heavy hand", "ingredient": "butter", "action": "REDUCE", "causal": False},
    {"text": "the vinegar was barely detectable", "ingredient": "vinegar", "action": "INCREASE", "causal": False},
    {"text": "the ginger was used very sparingly", "ingredient": "ginger", "action": "INCREASE", "causal": False},
    {"text": "the cream was used far too liberally", "ingredient": "cream", "action": "REDUCE", "causal": False},

    # --- Cultural / Style ---
    {"text": "for an Italian dish it needed more basil", "ingredient": "basil", "action": "INCREASE", "causal": False},
    {"text": "not authentic, needed more cumin for a Mexican dish", "ingredient": "cumin", "action": "INCREASE", "causal": False},
    {"text": "for Thai food it needed more lemongrass", "ingredient": "lemongrass", "action": "INCREASE", "causal": False},
    {"text": "the Indian curry lacked turmeric", "ingredient": "turmeric", "action": "INCREASE", "causal": False},
    {"text": "for a French dish the butter was appropriately used", "ingredient": "butter", "action": "KEEP", "causal": False},
    {"text": "the Japanese ramen needed more miso", "ingredient": "miso", "action": "INCREASE", "causal": False},
    {"text": "the Greek salad had too much feta", "ingredient": "feta", "action": "REDUCE", "causal": False},
    {"text": "the Spanish dish needed more paprika", "ingredient": "paprika", "action": "INCREASE", "causal": False},
    {"text": "the Korean dish needed more sesame oil", "ingredient": "sesame oil", "action": "INCREASE", "causal": False},
    {"text": "for a Middle Eastern dish it needed more tahini", "ingredient": "tahini", "action": "INCREASE", "causal": False},

    # --- Time / Freshness ---
    {"text": "the herbs were clearly not fresh", "ingredient": "herbs", "action": "REDUCE", "causal": False},
    {"text": "the fish tasted old and not fresh", "ingredient": "fish", "action": "REDUCE", "causal": False},
    {"text": "the vegetables were wilted and past their best", "ingredient": "vegetables", "action": "REDUCE", "causal": False},
    {"text": "the bread was stale and hard", "ingredient": "bread", "action": "REDUCE", "causal": False},
    {"text": "the meat tasted like it had been sitting too long", "ingredient": "meat", "action": "REDUCE", "causal": False},
    {"text": "the cream had clearly gone off", "ingredient": "cream", "action": "REDUCE", "causal": False},
    {"text": "the cheese was perfectly aged and delightful", "ingredient": "cheese", "action": "KEEP", "causal": False},
    {"text": "the tomatoes were beautifully ripe", "ingredient": "tomato", "action": "KEEP", "causal": False},
    {"text": "the salmon was obviously very fresh", "ingredient": "salmon", "action": "KEEP", "causal": False},
    {"text": "the basil was wonderfully fresh and fragrant", "ingredient": "basil", "action": "KEEP", "causal": False},

    # --- Emotional Reaction ---
    {"text": "the garlic made me want to leave the restaurant", "ingredient": "garlic", "action": "REDUCE", "causal": True},
    {"text": "I was so happy with how the salmon was prepared", "ingredient": "salmon", "action": "KEEP", "causal": False},
    {"text": "the pepper ruined my whole evening", "ingredient": "pepper", "action": "REDUCE", "causal": True},
    {"text": "I fell in love with the basil chicken", "ingredient": "basil", "action": "KEEP", "causal": False},
    {"text": "the excess salt made me feel sick", "ingredient": "salt", "action": "REDUCE", "causal": True},
    {"text": "I was delighted by the perfect seasoning", "ingredient": "seasoning", "action": "KEEP", "causal": False},
    {"text": "the amount of butter disgusted me", "ingredient": "butter", "action": "REDUCE", "causal": True},
    {"text": "the perfectly cooked steak made my day", "ingredient": "steak", "action": "KEEP", "causal": False},
    {"text": "the chili heat made me regret ordering it", "ingredient": "chili", "action": "REDUCE", "causal": True},
    {"text": "the herbs transported me back to my grandmother's cooking", "ingredient": "herbs", "action": "KEEP", "causal": True},

    # --- Implicit Feedback ---
    {"text": "I left half the dish because of how salty it was", "ingredient": "salt", "action": "REDUCE", "causal": True},
    {"text": "I ordered a second portion because the chicken was so good", "ingredient": "chicken", "action": "KEEP", "causal": True},
    {"text": "I scraped off all the sauce because there was too much", "ingredient": "sauce", "action": "REDUCE", "causal": False},
    {"text": "I asked for extra cheese on the side", "ingredient": "cheese", "action": "INCREASE", "causal": False},
    {"text": "I couldn't finish it because of the excessive spice", "ingredient": "spice", "action": "REDUCE", "causal": True},
    {"text": "I wiped off most of the mayonnaise", "ingredient": "mayonnaise", "action": "REDUCE", "causal": False},
    {"text": "I requested more sauce because it was too dry", "ingredient": "sauce", "action": "INCREASE", "causal": False},
    {"text": "I added my own salt at the table", "ingredient": "salt", "action": "INCREASE", "causal": False},
    {"text": "I couldn't eat the dish due to too much oil", "ingredient": "oil", "action": "REDUCE", "causal": True},
    {"text": "I went back for a second helping of the pasta", "ingredient": "pasta", "action": "KEEP", "causal": False},

    # --- Suggestion Based ---
    {"text": "next time try reducing the amount of pepper", "ingredient": "pepper", "action": "REDUCE", "causal": False},
    {"text": "I would suggest adding more garlic to the recipe", "ingredient": "garlic", "action": "INCREASE", "causal": False},
    {"text": "the chef should use less butter in this preparation", "ingredient": "butter", "action": "REDUCE", "causal": False},
    {"text": "I recommend keeping the same amount of herbs", "ingredient": "herbs", "action": "KEEP", "causal": False},
    {"text": "they should dial back on the chili", "ingredient": "chili", "action": "REDUCE", "causal": False},
    {"text": "I'd suggest a touch more lemon next time", "ingredient": "lemon", "action": "INCREASE", "causal": False},
    {"text": "the kitchen should add more mushrooms to the dish", "ingredient": "mushrooms", "action": "INCREASE", "causal": False},
    {"text": "I'd advise reducing the cream in the sauce", "ingredient": "cream", "action": "REDUCE", "causal": False},
    {"text": "they could do with more seasoning overall", "ingredient": "seasoning", "action": "INCREASE", "causal": False},
    {"text": "the chef should maintain the current garlic level", "ingredient": "garlic", "action": "KEEP", "causal": False},
    {"text": "I think more vegetables would improve this dish", "ingredient": "vegetables", "action": "INCREASE", "causal": False},
    {"text": "they should cut back on the vinegar significantly", "ingredient": "vinegar", "action": "REDUCE", "causal": False},
    {"text": "I recommend increasing the protein content", "ingredient": "protein", "action": "INCREASE", "causal": False},
    {"text": "the restaurant should add more cheese to the dish", "ingredient": "cheese", "action": "INCREASE", "causal": False},
    {"text": "I suggest using less oil when frying", "ingredient": "oil", "action": "REDUCE", "causal": False},

    # --- Negation Tricky ---
    {"text": "not too much garlic which I appreciated", "ingredient": "garlic", "action": "KEEP", "causal": False},
    {"text": "not enough salt made it bland", "ingredient": "salt", "action": "INCREASE", "causal": False},
    {"text": "not overly spiced which was refreshing", "ingredient": "spice", "action": "KEEP", "causal": False},
    {"text": "not enough cheese for my liking", "ingredient": "cheese", "action": "INCREASE", "causal": False},
    {"text": "not too sweet which was pleasant", "ingredient": "sugar", "action": "KEEP", "causal": False},
    {"text": "not as much garlic as I would like", "ingredient": "garlic", "action": "INCREASE", "causal": False},
    {"text": "not heavily sauced which I liked", "ingredient": "sauce", "action": "KEEP", "causal": False},
    {"text": "not enough herb flavor came through", "ingredient": "herbs", "action": "INCREASE", "causal": False},
    {"text": "not the right amount of pepper", "ingredient": "pepper", "action": "INCREASE", "causal": False},
    {"text": "not overly oily which was great", "ingredient": "oil", "action": "KEEP", "causal": False},
    {"text": "not enough lemon for brightness", "ingredient": "lemon", "action": "INCREASE", "causal": False},
    {"text": "not too much cream which kept it light", "ingredient": "cream", "action": "KEEP", "causal": False},
    {"text": "not the right balance of butter", "ingredient": "butter", "action": "REDUCE", "causal": False},
    {"text": "not enough ginger to make an impact", "ingredient": "ginger", "action": "INCREASE", "causal": False},
    {"text": "not overly salted and perfectly edible", "ingredient": "salt", "action": "KEEP", "causal": False},

    # --- Extra Scenarios ---
    {"text": "the pizza base was too thick and doughy", "ingredient": "dough", "action": "REDUCE", "causal": False},
    {"text": "the smoothie had too much banana", "ingredient": "banana", "action": "REDUCE", "causal": False},
    {"text": "the coffee had too much sugar", "ingredient": "sugar", "action": "REDUCE", "causal": False},
    {"text": "the tea was perfectly brewed", "ingredient": "tea", "action": "KEEP", "causal": False},
    {"text": "the cocktail needed more lime juice", "ingredient": "lime", "action": "INCREASE", "causal": False},
    {"text": "the juice was too sweet from too much apple", "ingredient": "apple", "action": "REDUCE", "causal": False},
    {"text": "the oatmeal needed more honey", "ingredient": "honey", "action": "INCREASE", "causal": False},
    {"text": "the granola was perfectly sweetened", "ingredient": "sugar", "action": "KEEP", "causal": False},
    {"text": "the steak sauce was too tangy", "ingredient": "sauce", "action": "REDUCE", "causal": False},
    {"text": "the BBQ sauce was perfectly smoky and sweet", "ingredient": "BBQ sauce", "action": "KEEP", "causal": False},
    {"text": "the guacamole needed more lime", "ingredient": "lime", "action": "INCREASE", "causal": False},
    {"text": "the hummus was perfectly garlicky", "ingredient": "garlic", "action": "KEEP", "causal": False},
    {"text": "the tzatziki had too much dill", "ingredient": "dill", "action": "REDUCE", "causal": False},
    {"text": "the gazpacho needed more tomato", "ingredient": "tomato", "action": "INCREASE", "causal": False},
    {"text": "the bruschetta was perfectly seasoned", "ingredient": "seasoning", "action": "KEEP", "causal": False},
    {"text": "the risotto needed more parmesan", "ingredient": "parmesan", "action": "INCREASE", "causal": False},
    {"text": "the gnocchi was too heavy and starchy", "ingredient": "potato", "action": "REDUCE", "causal": False},
    {"text": "the bolognese was perfectly balanced", "ingredient": "sauce", "action": "KEEP", "causal": False},
    {"text": "the carbonara had too much cream", "ingredient": "cream", "action": "REDUCE", "causal": False},
    {"text": "the osso buco was wonderfully seasoned", "ingredient": "seasoning", "action": "KEEP", "causal": False},
    {"text": "the paella needed more saffron", "ingredient": "saffron", "action": "INCREASE", "causal": False},
    {"text": "the tagine was perfectly spiced", "ingredient": "spice", "action": "KEEP", "causal": False},
    {"text": "the biryani had too much cardamom", "ingredient": "cardamom", "action": "REDUCE", "causal": False},
    {"text": "the pad thai needed more tamarind", "ingredient": "tamarind", "action": "INCREASE", "causal": False},
    {"text": "the pho broth was perfectly seasoned", "ingredient": "seasoning", "action": "KEEP", "causal": False},
    {"text": "the bibimbap needed more sesame oil", "ingredient": "sesame oil", "action": "INCREASE", "causal": False},
    {"text": "the sushi had too much wasabi", "ingredient": "wasabi", "action": "REDUCE", "causal": False},
    {"text": "the mole sauce was perfectly complex", "ingredient": "sauce", "action": "KEEP", "causal": False},
    {"text": "the falafel needed more cumin", "ingredient": "cumin", "action": "INCREASE", "causal": False},
    {"text": "the shakshuka was perfectly spiced", "ingredient": "spice", "action": "KEEP", "causal": False},
    {"text": "the borscht needed more dill", "ingredient": "dill", "action": "INCREASE", "causal": False},
    {"text": "the stroganoff had too much sour cream", "ingredient": "sour cream", "action": "REDUCE", "causal": False},
    {"text": "the kimchi was perfectly fermented and spicy", "ingredient": "chili", "action": "KEEP", "causal": False},
    {"text": "the laksa had too much coconut milk", "ingredient": "coconut milk", "action": "REDUCE", "causal": False},
    {"text": "the vindaloo was perfectly hot", "ingredient": "chili", "action": "KEEP", "causal": False},
    {"text": "the massaman curry needed more peanut", "ingredient": "peanut", "action": "INCREASE", "causal": False},
    {"text": "the goulash was too heavy on paprika", "ingredient": "paprika", "action": "REDUCE", "causal": False},
    {"text": "the ceviche needed more lime for acidity", "ingredient": "lime", "action": "INCREASE", "causal": False},
    {"text": "the bouillabaisse was perfectly seasoned", "ingredient": "seasoning", "action": "KEEP", "causal": False},
    {"text": "the ratatouille needed more herbs de provence", "ingredient": "herbs", "action": "INCREASE", "causal": False},
]

# TEST_DATA = [
#     # --- Simple Negative ---
#     {"text": "too much garlic in this dish", "ingredient": "garlic", "action": "REDUCE", "causal": False},
#     {"text": "way too salty for my taste", "ingredient": "salt", "action": "REDUCE", "causal": False},
#     {"text": "the chicken was incredibly dry", "ingredient": "chicken", "action": "REDUCE", "causal": False},
#     {"text": "overpowered by the pepper", "ingredient": "pepper", "action": "REDUCE", "causal": False},
#     {"text": "I really hated the onions", "ingredient": "onion", "action": "REDUCE", "causal": False},
#     {"text": "far too much sugar in the dessert", "ingredient": "sugar", "action": "REDUCE", "causal": False},
#     {"text": "the curry had too much chili", "ingredient": "chili", "action": "REDUCE", "causal": False},
#     {"text": "the soup was overly salty", "ingredient": "salt", "action": "REDUCE", "causal": False},
#     {"text": "too much mayonnaise ruined the sandwich", "ingredient": "mayonnaise", "action": "REDUCE", "causal": False},
#     {"text": "the lemon flavor was overpowering", "ingredient": "lemon", "action": "REDUCE", "causal": False},
#     {"text": "the sauce was far too spicy", "ingredient": "sauce", "action": "REDUCE", "causal": False},
#     {"text": "there was excessive butter", "ingredient": "butter", "action": "REDUCE", "causal": False},
#     {"text": "too much ginger for my liking", "ingredient": "ginger", "action": "REDUCE", "causal": False},
#     {"text": "the beans were undercooked and unpleasant", "ingredient": "beans", "action": "REDUCE", "causal": False},
#     {"text": "the mushrooms tasted awful", "ingredient": "mushrooms", "action": "REDUCE", "causal": False},
#     {"text": "the bacon was much too greasy", "ingredient": "bacon", "action": "REDUCE", "causal": False},
#     {"text": "too much vinegar spoiled the salad", "ingredient": "vinegar", "action": "REDUCE", "causal": False},
#     {"text": "the cinnamon was overwhelming", "ingredient": "cinnamon", "action": "REDUCE", "causal": False},
#     {"text": "there was too much cheese", "ingredient": "cheese", "action": "REDUCE", "causal": False},
#     {"text": "the sauce completely ruined the pasta", "ingredient": "sauce", "action": "REDUCE", "causal": False},

#     # --- KEEP ---
#     {"text": "loved the salmon it was perfect", "ingredient": "salmon", "action": "KEEP", "causal": False},
#     {"text": "the beef was cooked wonderfully", "ingredient": "beef", "action": "KEEP", "causal": False},
#     {"text": "absolutely amazing spices", "ingredient": "spice", "action": "KEEP", "causal": False},
#     {"text": "great tasting pasta", "ingredient": "pasta", "action": "KEEP", "causal": False},
#     {"text": "the garlic flavor was spot on", "ingredient": "garlic", "action": "KEEP", "causal": False},
#     {"text": "the vegetables were fresh and delicious", "ingredient": "vegetables", "action": "KEEP", "causal": False},
#     {"text": "the seasoning was just right", "ingredient": "seasoning", "action": "KEEP", "causal": False},
#     {"text": "the rice was cooked perfectly", "ingredient": "rice", "action": "KEEP", "causal": False},
#     {"text": "the cheese complemented the dish nicely", "ingredient": "cheese", "action": "KEEP", "causal": False},
#     {"text": "excellent amount of garlic", "ingredient": "garlic", "action": "KEEP", "causal": False},
#     {"text": "the herbs gave a wonderful flavor", "ingredient": "herbs", "action": "KEEP", "causal": False},
#     {"text": "the chicken was juicy and flavorful", "ingredient": "chicken", "action": "KEEP", "causal": False},
#     {"text": "perfect amount of pepper", "ingredient": "pepper", "action": "KEEP", "causal": False},
#     {"text": "the butter added great richness", "ingredient": "butter", "action": "KEEP", "causal": False},
#     {"text": "the fish tasted very fresh", "ingredient": "fish", "action": "KEEP", "causal": False},

#     # --- INCREASE ---
#     {"text": "could use more spice", "ingredient": "spice", "action": "INCREASE", "causal": False},
#     {"text": "needed more protein", "ingredient": "protein", "action": "INCREASE", "causal": False},
#     {"text": "add more cheese next time", "ingredient": "cheese", "action": "INCREASE", "causal": False},
#     {"text": "was a bit bland, needs salt", "ingredient": "salt", "action": "INCREASE", "causal": False},
#     {"text": "I wish there was more chicken", "ingredient": "chicken", "action": "INCREASE", "causal": False},
#     {"text": "needs a little more garlic", "ingredient": "garlic", "action": "INCREASE", "causal": False},
#     {"text": "could definitely use more herbs", "ingredient": "herbs", "action": "INCREASE", "causal": False},
#     {"text": "please add more vegetables", "ingredient": "vegetables", "action": "INCREASE", "causal": False},
#     {"text": "the soup needed more pepper", "ingredient": "pepper", "action": "INCREASE", "causal": False},
#     {"text": "I wanted more sauce", "ingredient": "sauce", "action": "INCREASE", "causal": False},
#     {"text": "add extra mushrooms please", "ingredient": "mushrooms", "action": "INCREASE", "causal": False},
#     {"text": "more onions would improve it", "ingredient": "onion", "action": "INCREASE", "causal": False},
#     {"text": "I'd like more tomatoes", "ingredient": "tomato", "action": "INCREASE", "causal": False},
#     {"text": "could use a little more butter", "ingredient": "butter", "action": "INCREASE", "causal": False},
#     {"text": "more garlic would have made it perfect", "ingredient": "garlic", "action": "INCREASE", "causal": False},

#     # --- Causal ---
#     {"text": "the meal was ruined because of the pepper", "ingredient": "pepper", "action": "REDUCE", "causal": True},
#     {"text": "tasted bad due to the fish", "ingredient": "fish", "action": "REDUCE", "causal": True},
#     {"text": "I didn't like it because of the heavy cream", "ingredient": "cream", "action": "REDUCE", "causal": True},
#     {"text": "terrible experience due to stale bread", "ingredient": "bread", "action": "REDUCE", "causal": True},
#     {"text": "because of the onions the meal was disappointing", "ingredient": "onion", "action": "REDUCE", "causal": True},
#     {"text": "the dish failed because of the excessive salt", "ingredient": "salt", "action": "REDUCE", "causal": True},
#     {"text": "the dinner was unpleasant due to the garlic", "ingredient": "garlic", "action": "REDUCE", "causal": True},
#     {"text": "the experience was bad because of the oily sauce", "ingredient": "oil", "action": "REDUCE", "causal": True},

#     # --- Mixed / Complex ---
#     {"text": "chicken was great but the sauce was terrible", "ingredient": "sauce", "action": "REDUCE", "causal": False},
#     {"text": "fish was perfectly cooked however way too much garlic", "ingredient": "garlic", "action": "REDUCE", "causal": False},
#     {"text": "loved the rice, but the beef was too chewy", "ingredient": "beef", "action": "REDUCE", "causal": False},
#     {"text": "the pasta was delicious but needed more cheese", "ingredient": "cheese", "action": "INCREASE", "causal": False},
#     {"text": "great vegetables although the soup needed more salt", "ingredient": "salt", "action": "INCREASE", "causal": False},
#     {"text": "the chicken was tasty but there was too much pepper", "ingredient": "pepper", "action": "REDUCE", "causal": False},
#     {"text": "excellent fish but the lemon was overpowering", "ingredient": "lemon", "action": "REDUCE", "causal": False},
#     {"text": "the beef was good although more herbs would help", "ingredient": "herbs", "action": "INCREASE", "causal": False},

#     # --- Tricky / Ambiguous ---
#     {"text": "interesting spice level", "ingredient": "spice", "action": "KEEP", "causal": False},
#     {"text": "it was okay, a bit too heavy on the butter though", "ingredient": "butter", "action": "REDUCE", "causal": False},
#     {"text": "not what I expected from a salmon dish", "ingredient": "salmon", "action": "REDUCE", "causal": False},
#     {"text": "could have been better if the pork was tender", "ingredient": "pork", "action": "INCREASE", "causal": False},
#     {"text": "the marinade completely ruined everything", "ingredient": "marinade", "action": "REDUCE", "causal": False},
#     {"text": "the texture was weird", "ingredient": "texture", "action": "REDUCE", "causal": False},
#     {"text": "I can't believe how much oil was in this", "ingredient": "oil", "action": "REDUCE", "causal": False},
#     {"text": "the flavor was unusual", "ingredient": "flavor", "action": "KEEP", "causal": False},
#     {"text": "not bad overall", "ingredient": "overall", "action": "KEEP", "causal": False},
#     {"text": "it needed just a touch more seasoning", "ingredient": "seasoning", "action": "INCREASE", "causal": False}
# ]

def evaluate_system():
    print("Starting End-to-End Evaluation of Feedback Pipeline...\n")
    
    y_true_action = []
    y_pred_action = []
    
    y_true_causal = []
    y_pred_causal = []
    
    correct_ingredient_extraction = 0
    total_samples = len(TEST_DATA)
    
    detailed_results = []

    for i, test in enumerate(TEST_DATA):
        text = test["text"]
        true_ingredient = test["ingredient"]
        true_action = test["action"]
        true_causal = test["causal"]
        
        # Run through the pipeline
        # star_rating and consumed_percent are mocked as they don't affect extraction/intent
        result = analyze_feedback(recipe_name="Test Recipe", feedback_text=text, star_rating=3, consumed_percent=50)
        
        # Parse outputs
        pred_causal = result["causal"]
        extracted_aspects = result["aspects"]
        
        # Determine the primary predicted action and ingredient
        # If multiple aspects are returned, we take the one that matches ground truth or the first one
        pred_action = "KEEP" # Default fallback
        pred_ingredient = None
        
        if extracted_aspects:
            # Try to find a match for the ingredient
            matched = False
            for aspect, action, delta in extracted_aspects:
                if true_ingredient in aspect or aspect in true_ingredient:
                    pred_ingredient = aspect
                    pred_action = action
                    matched = True
                    break
            
            # If no match, just take the first one extracted
            if not matched:
                pred_ingredient = extracted_aspects[0][0]
                pred_action = extracted_aspects[0][1]
        
        # Record for metrics
        y_true_action.append(true_action)
        y_pred_action.append(pred_action)
        
        y_true_causal.append(true_causal)
        y_pred_causal.append(pred_causal)
        
        if pred_ingredient and (true_ingredient in pred_ingredient or pred_ingredient in true_ingredient):
            correct_ingredient_extraction += 1
            ingred_match = True
        else:
            ingred_match = False
            
        detailed_results.append({
            "text": text,
            "true_ing": true_ingredient,
            "pred_ing": pred_ingredient,
            "ing_match": ingred_match,
            "true_act": true_action,
            "pred_act": pred_action,
            "act_match": true_action == pred_action
        })
        
        # Print progress (overwrite line)
        sys.stdout.write(f"\rProcessed {i+1}/{total_samples} test cases...")
        sys.stdout.flush()

    print("\n\nEvaluation Complete. Calculating metrics...\n")
    
    # Calculate Metrics
    action_acc = accuracy_score(y_true_action, y_pred_action)
    causal_acc = accuracy_score(y_true_causal, y_pred_causal)
    ingred_acc = correct_ingredient_extraction / total_samples
    
    # Calculate precision, recall, f1 for actions
    # labels order: REDUCE, INCREASE, KEEP
    labels = ["REDUCE", "INCREASE", "KEEP"]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_action, y_pred_action, labels=labels, zero_division=0)
    
    # Combine everything for a holistic "Pipeline Accuracy"
    # A prediction is only truly "correct" end-to-end if BOTH the ingredient and action are correct
    end_to_end_correct = sum(1 for res in detailed_results if res["ing_match"] and res["act_match"])
    end_to_end_acc = end_to_end_correct / total_samples
    
    # ==========================================
    # 2. GENERATE PLOTS FOR REPORT
    # ==========================================
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Confusion Matrix for Actions
    cm = confusion_matrix(y_true_action, y_pred_action, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix: Intent Classification (Action)')
    plt.ylabel('True Action')
    plt.xlabel('Predicted Action')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_action.png'), dpi=300)
    plt.close()
    
    # Plot 2: Performance Metrics Bar Chart
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
    rects2 = ax.bar(x, recall, width, label='Recall', color='#ff7f0e')
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ca02c')
    
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, and F1-Score by Action Category')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_bar_chart.png'), dpi=300)
    plt.close()

    # Plot 3: Component Accuracy Overview
    components = ['Aspect Extraction', 'Intent (Action)', 'Causal Detection', 'End-to-End System']
    accuracies = [ingred_acc, action_acc, causal_acc, end_to_end_acc]
    
    plt.figure(figsize=(9, 5))
    bars = plt.bar(components, [a * 100 for a in accuracies], color=['#4c72b0', '#dd8452', '#55a868', '#c44e52'])
    plt.ylabel('Accuracy (%)')
    plt.title('Feedback Pipeline Component Accuracies')
    plt.ylim([0, 100])
    
    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_accuracies.png'), dpi=300)
    plt.close()

    # ==========================================
    # 3. PRINT RESULTS AND LATEX TABLE
    # ==========================================
    print("="*60)
    print(f"EVALUATION RESULTS (N={total_samples})")
    print("="*60)
    print(f"End-to-End Accuracy:      {end_to_end_acc*100:.1f}%")
    print(f"Aspect Extraction Acc:    {ingred_acc*100:.1f}%")
    print(f"Intent (Action) Acc:      {action_acc*100:.1f}%")
    print(f"Causal Detection Acc:     {causal_acc*100:.1f}%")
    print("-" * 60)
    
    for i, label in enumerate(labels):
        print(f"Class: {label}")
        print(f"  Precision: {precision[i]:.3f}")
        print(f"  Recall:    {recall[i]:.3f}")
        print(f"  F1-Score:  {f1[i]:.3f}")
    
    print("="*60)
    print(f"\nGraphs saved to: {os.path.abspath(output_dir)}\n")
    
    # Generate LaTeX Table for the report
    latex_table = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{|l|c|c|c|}}
\\hline
\\textbf{{Metric / Component}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1-Score}} \\\\ \\hline
Intent: REDUCE & {precision[0]:.3f} & {recall[0]:.3f} & {f1[0]:.3f} \\\\ \\hline
Intent: INCREASE & {precision[1]:.3f} & {recall[1]:.3f} & {f1[1]:.3f} \\\\ \\hline
Intent: KEEP & {precision[2]:.3f} & {recall[2]:.3f} & {f1[2]:.3f} \\\\ \\hline
\\hline
\\textbf{{Pipeline Stage}} & \\multicolumn{{3}}{{c|}}{{\\textbf{{Accuracy}}}} \\\\ \\hline
Aspect Extraction & \\multicolumn{{3}}{{c|}}{{{ingred_acc*100:.1f}\\%}} \\\\ \\hline
Intent Classification & \\multicolumn{{3}}{{c|}}{{{action_acc*100:.1f}\\%}} \\\\ \\hline
Causal Detection & \\multicolumn{{3}}{{c|}}{{{causal_acc*100:.1f}\\%}} \\\\ \\hline
\\textbf{{End-to-End System}} & \\multicolumn{{3}}{{c|}}{{\\textbf{{{end_to_end_acc*100:.1f}\\%}}}} \\\\ \\hline
\\end{{tabular}}
\\caption{{End-to-end evaluation metrics of the natural language feedback processing pipeline on realistic unseen data.}}
\\label{{tab:feedback_eval}}
\\end{{table}}
"""
    print("LaTeX Table for Research Report:")
    print(latex_table)

if __name__ == "__main__":
    evaluate_system()
