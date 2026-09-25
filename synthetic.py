from __future__ import annotations

import argparse
import json
import random
import uuid

from constants import SYNTHETIC_DATA


def _fct_dict():
    return {
        n: rng_attr(n) for n in _NATIONS
    }


def rng_attr(nation):
    return {
        "capital city": f"Port {nation}",
        "official language": f"{nation}ish",
        "river": f"the {nation} River",
        "national animal": f"the {nation} fox",
        "national flower": f"the {nation} lily",
        "currency": f"the {nation} crown",
    }


_NATIONS = [
    "Fictionalia", "Arandor", "Boronia", "Caldenia", "Delmark", "Esterfall",
    "Farrador", "Glenrock", "Holmstead", "Iverlund", "Jovandria", "Kestrel Bay",
    "Lankovar", "Marrowgale", "Nyspeth", "Orandel", "Pellmoor", "Quenstead",
    "Ravencross", "Sallowmere", "Tarnhelm", "Umbraden", "Vandermark", "Wistmoor",
    "Xanthol", "Yarmond", "Zarovna", "Asterfell", "Brightcoast", "Cadence Vale",
    "Driftwood", "Embermarch", "Forsythe", "Garland Reach", "Harvestglen",
    "Ironwych", "Juniper Tor", "Kestwood", "Liddestone", "Moonhaven",
    "Northcrag", "Oakhallow", "Prismere", "Quicksilver Bay", "Ridgemont",
    "Stonemark", "Thistlewick", "Umberland",
]

_FACTS = _fct_dict()

_RETRIEVAL_PHRASES = [
    "The {attr} of {obj} is {value}.",
    "According to the {obj} registry, the {attr} is {value}.",
    "In the records for {obj}, we find that the {attr} is {value}.",
    "Officials in {obj} state that its {attr} is {value}.",
    "The official {attr} of {obj} is listed as {value}.",
    "An atlas entry for {obj} records its {attr} as {value}.",
    "The survey notes that {obj} has {attr} of {value}.",
    "By treaty, the {attr} of {obj} is {value}.",
    "The {obj} almanac names its {attr} as {value}.",
    "A profile of {obj} indicates that {attr} is {value}.",
    "Records indicate that {obj} set its {attr} at {value}.",
    "The constitution of {obj} fixes its {attr} at {value}.",
]

_NAMES = [
    "Ana", "Ben", "Cal", "Dan", "Eve", "Fia", "Gil", "Hal",
    "Ivo", "Jax", "Kim", "Leo", "Mia", "Noa", "Ola", "Pia",
    "Quin", "Rex", "Sue", "Tia", "Ugo", "Vic", "Wyn", "Xia",
    "Yves", "Zed", "Ari", "Blaise", "Cora", "Dawn", "Ellis", "Finn",
    "Greta", "Hugo", "Ida", "Jules", "Kira", "Lars", "Mona", "Nell",
    "Oren", "Pat", "Ruth", "Sami", "Tess", "Uma", "Vera", "Ward",
]

_EVENTS = [
    "the package was delivered", "the customer emailed support",
    "the refund was issued", "the courier was called",
    "the manager approved it", "the invoice was sent",
    "the item was restocked", "the complaint was filed",
    "the payment was processed", "the address was changed",
    "the box was scanned", "the warehouse was notified",
    "the review was published", "the tracking number was shared",
    "the warranty was registered", "the return label was printed",
    "the shipment departed", "the customer was notified",
    "the receipt was emailed", "the stock count was updated",
    "the label was rescanned", "the courier handed over the parcel",
    "the packing slip was attached", "the pallet was sealed",
    "the dock door opened", "the driver confirmed pickup",
    "the invoice was settled", "the claim was processed",
    "the manifest was signed", "the order was archived",
    "the item was quality checked", "the batch number was recorded",
    "the shelf was restocked", "the supplier was invoiced",
    "the discount was applied", "the gift note was added",
    "the packaging was recycled", "the barcode was reissued",
    "the complaint was escalated", "the ticket was closed",
    "the sample was mailed", "the test was approved",
    "the account was credited", "the hold was lifted",
    "the route was changed", "the delivery window was reserved",
    "the coupon was redeemed", "the news was broadcast",
]

_ORDER_PHRASES = [
    "First {a}, then {b}. Afterwards {c}.",
    "To begin, {a}. Next came {b}, and finally {c}.",
    "{a} happened first, followed by {b}, and then {c}.",
    "The order was: {a}, then {b}, and last of all {c}.",
    "In sequence: {a} first, {b} second, {c} third.",
    "Step one {a}; after that {b}; concluding with {c}.",
]

_INTENT = {
    "return": ["wants to return the product", "is asking to send it back", "requested a refund", "would like her money back", "hopes to get a refund for the item"],
    "replace": ["wants a replacement unit", "asked for a new copy of the item", "needs the product exchanged", "hopes to trade it in for a fresh one", "would like a replacement device"],
    "support": ["needs help setting it up", "asked how to install it", "wants setup assistance", "is stuck during configuration", "needs a step by step guide"],
    "cancel": ["wants to cancel the order", "asked to stop the delivery", "hopes to withdraw the purchase", "would like the order cancelled", "wants to back out of the purchase"],
    "verify": ["wants confirmation of delivery", "asked whether the package arrived", "needs proof it was shipped", "wants to confirm the tracking number", "asks for delivery verification"],
    "compliment": ["wants to praise the service", "liked how fast it shipped", "wants to leave a positive note", "is happy with the purchase", "would like to compliment the staff"],
    "negotiate": ["wants a lower price", "asked for a discount", "hopes to haggle over the cost", "wants to negotiate the fee", "would like a better rate"],
    "upgrade": ["wants a newer model", "asked for the latest version", "hopes to upgrade the plan", "wants more capacity", "would like an upgraded membership"],
}

_INTENT_LABEL = {
    "return": "issue a refund",
    "replace": "send a replacement",
    "support": "provide setup help",
    "cancel": "cancel the order",
    "verify": "confirm the delivery",
    "compliment": "acknowledge a compliment",
    "negotiate": "discuss the price",
    "upgrade": "offer an upgrade",
}

_AMBIGUITY_PH = [
    "The lights on the third floor flickered twice and the printer stopped. Nobody reported anything unusual.",
    "The front door was left unlocked again and the coffee maker was still warm. No one else has a key.",
    "The report file was changed at 3am and the backup ran at 3:30am. Only the night shift was in the building.",
    "The sensor logged a spike at dawn and the alarm sounded moments later. There is no camera footage for that window.",
    "The weekend server reboot succeeded, but by Monday morning every login token had expired. The ops log shows no manual action.",
    "The lift car creaked to a halt between floors and the intercom played a two-tone chime. The duty roster lists no technician on site.",
    "The gate opened at dusk even though the schedule said it was locked. The access log shows a badge that was reported lost last Thursday.",
    "The irrigation valve turned on overnight although the controller was set to off. A repair record from a month ago mentioned a loose relay.",
    "The projector shut off mid-meeting and the room lights dimmed for a second before returning. No one touched the panel.",
    "The mailbox was emptied before sunrise and the parcel you ordered was inside. The mail carrier usually passes after nine.",
    "The alarm keypad beeped once at 2:14am and again at 2:15am, then fell silent. The door sensors show no motion.",
    "The backup tape was found outside the safe, rewound to its start. The nightly job ran without errors.",
    "The water pressure dropped for five minutes then returned. The maintenance log mentions work on the street mains this month.",
    "The cash register drawer opened on its own at closing time. The security review found no refunds or voids processed.",
    "The greenhouse temperature hit 40 degrees overnight. The thermostat was set to 22 and the heater is electric.",
    "The archive server froze and the clock reset to midnight on restart. The uninterruptible supply logged a sag at 1:07am.",
]

_AMB_OPTIONS = [
    "a brief power fluctuation", "an untracked employee action",
    "a failing peripheral", "a scheduled maintenance step",
    "an error in the logging system", "a prank or accident",
    "a delayed automatic process", "an outside network event",
]

_AMB_PROBS = [0.34, 0.14, 0.09, 0.19, 0.08, 0.05, 0.08, 0.03]

N_OPTIONS = 8

_SPATIAL_PLACES = {
    "shelf": ["mug", "cup", "jar", "bowl", "dish", "plate", "tin", "can", "bottle", "box", "flask", "crock"],
    "desk row": ["lamp", "pen", "eraser", "ruler", "cup", "book", "clip", "notebook", "stamp", "tape", "compass", "pencil"],
    "hook row": ["jacket", "coat", "scarf", "hat", "gloves", "umbrella", "bag", "cap", "scarf", "vest", "shawl", "coat"],
    "workbench": ["hammer", "wrench", "pliers", "saw", "chisel", "drill", "file", "vise", "mallet", "plane", "rasp", "screwdriver"],
    "counter": ["kettle", "toaster", "blender", "scale", "jar", "spice rack", "cutting board", "mixing bowl", "colander", "whisk", "tray", "mortar"],
    "bookshelf": ["novel", "atlas", "dictionary", "journal", "manual", "yearbook", "poetry", "biography", "encyclopedia", "album", "reader", "cookbook"],
    "pantry shelf": ["salt", "rice", "oats", "honey", "beans", "lentils", "flour", "tea", "oil", "sugar", "cocoa", "cereal"],
    "toolshed wall": ["shovel", "rake", "hoe", "scythe", "trowel", "pruner", "shears", "fork", "mattock", "spade", "edger", "auger"],
    "dressing table": ["comb", "brush", "mirror", "scissors", "pomade", "handkerchief", "button", "thimble", "nail file", "lotion", "clips", "perfume"],
    "hall cupboard": ["boots", "lantern", "torch", "goggles", "rope", "map", "whistle", "matches", "flask", "compass", "cord", "batteries"],
    "kitchen rack": ["ladle", "spatula", "tongs", "whisk", "ladle", "strainer", "peeler", "grater", "masher", "scoop", "roller", "brush"],
    "stationery tray": ["pencil", "ruler", "stapler", "hole punch", "tape", "scissors", "marker", "eraser", "sharpener", "glue", "pins", "clips"],
}

_COMPARE_ATTRS = [
    {"unit": "cm", "verb": "tall", "q": ("tallest", "shortest")},
    {"unit": "kg", "verb": "heavier", "q": ("heaviest", "lightest")},
    {"unit": "km/h", "verb": "fast", "q": ("fastest", "slowest")},
    {"unit": "eur", "verb": "costing", "q": ("most expensive", "least expensive")},
    {"unit": "years", "verb": "old", "q": ("oldest", "youngest")},
    {"unit": "kg", "verb": "laden with", "q": ("most heavily laden", "least heavily laden")},
    {"unit": "points", "verb": "scoring", "q": ("top score", "bottom score")},
    {"unit": "pages", "verb": "long", "q": ("longest", "shortest")},
]

_ODD_CATS = {
    "fruits": ["apple", "banana", "cherry", "date", "elderberry", "grape", "kiwi", "pear", "plum", "nectarine", "apricot", "fig"],
    "animals": ["dog", "cat", "horse", "cow", "sheep", "goat", "pig", "chicken", "duck", "goose", "turkey", "rabbit"],
    "tools": ["hammer", "wrench", "screwdriver", "pliers", "saw", "chisel", "drill", "file", "plane", "rasp", "mallet", "vise"],
    "metals": ["gold", "silver", "copper", "iron", "tin", "zinc", "nickel", "lead", "brass", "steel", "bronze", "aluminum"],
    "vegetables": ["carrot", "onion", "potato", "leek", "celery", "cabbage", "spinach", "kale", "turnip", "beet", "parsnip", "radish"],
    "planets": ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"],
    "instruments": ["piano", "violin", "guitar", "flute", "drum", "trumpet", "harp", "cello", "clarinet", "oboe", "trombone", "bassoon"],
    "birds": ["eagle", "sparrow", "finch", "robin", "crow", "dove", "swan", "heron", "owl", "hawk", "parrot", "wren"],
    "colors": ["red", "blue", "green", "yellow", "orange", "purple", "white", "black", "brown", "pink", "gray", "teal"],
    "sports": ["tennis", "soccer", "cricket", "rugby", "hockey", "basketball", "golf", "boxing", "cycling", "skating", "swimming", "rowing"],
    "musical genres": ["jazz", "blues", "rock", "folk", "pop", "reggae", "classical", "country", "metal", "hip hop", "disco", "soul"],
    "vehicles": ["car", "bus", "truck", "train", "boat", "plane", "bicycle", "helicopter", "motorcycle", "tram", "subway", "ferry"],
    "body parts": ["elbow", "knee", "wrist", "ankle", "neck", "hip", "shoulder", "toe", "finger", "heel", "chin", "temple"],
    "weather": ["rain", "snow", "hail", "sleet", "fog", "drizzle", "wind", "thunder", "blizzard", "mist", "sunshine", "frost"],
    "seas creatures": ["whale", "dolphin", "seal", "shark", "octopus", "jellyfish", "starfish", "crab", "lobster", "shrimp", "squid", "eel"],
    "buildings": ["lighthouse", "windmill", "castle", "church", "tower", "bridge", "barn", "shed", "palace", "fort", "temple", "monument"],
}

_CAT_CLASSES = ["mammal", "bird", "fish", "reptile", "amphibian", "insect", "crustacean", "mollusk"]
_CAT_MAP = {
    "dog": "mammal", "cat": "mammal", "horse": "mammal", "cow": "mammal", "whale": "mammal",
    "bat": "mammal", "kangaroo": "mammal", "elephant": "mammal", "deer": "mammal", "bear": "mammal",
    "wolf": "mammal", "fox": "mammal", "lion": "mammal", "tiger": "mammal", "bear": "mammal",
    "dolphin": "mammal", "seal": "mammal", "walrus": "mammal", "otter": "mammal", "beaver": "mammal",
    "rabbit": "mammal", "hare": "mammal", "mouse": "mammal", "rat": "mammal", "squirrel": "mammal",
    "chimpanzee": "mammal", "gorilla": "mammal", "orangutan": "mammal", "monkey": "mammal",
    "hippopotamus": "mammal", "rhinoceros": "mammal", "camel": "mammal", "giraffe": "mammal",
    "zebra": "mammal", "wildebeest": "mammal", "mole": "mammal", "shrew": "mammal", "hedgehog": "mammal",
    "eagle": "bird", "sparrow": "bird", "owl": "bird", "robin": "bird", "crow": "bird",
    "penguin": "bird", "swan": "bird", "ostrich": "bird", "hawk": "bird", "heron": "bird",
    "stork": "bird", "peacock": "bird", "parrot": "bird", "flamingo": "bird", "seagull": "bird",
    "pigeon": "bird", "woodpecker": "bird", "kingfisher": "bird", "wren": "bird", "finch": "bird",
    "vulture": "bird", "falcon": "bird", "duck": "bird", "goose": "bird", "chicken": "bird",
    "kestrel": "bird", "raven": "bird", "magpie": "bird", "jay": "bird", "starling": "bird",
    "salmon": "fish", "trout": "fish", "cod": "fish", "tuna": "fish", "mackerel": "fish",
    "bass": "fish", "herring": "fish", "sardine": "fish", "pike": "fish", "carp": "fish",
    "perch": "fish", "anchovy": "fish", "flounder": "fish", "halibut": "fish", "catfish": "fish",
    "eel": "fish", "grouper": "fish", "snapper": "fish", "tilapia": "fish", "goby": "fish",
    "snake": "reptile", "lizard": "reptile", "turtle": "reptile", "crocodile": "reptile",
    "gecko": "reptile", "iguana": "reptile", "chameleon": "reptile", "tortoise": "reptile",
    "alligator": "reptile", "caiman": "reptile", "monitor": "reptile", "newl": "reptile",
    "skink": "reptile", "mamba": "reptile", "python": "reptile", "viper": "reptile",
    "frog": "amphibian", "toad": "amphibian", "newt": "amphibian", "salamander": "amphibian",
    "axolotl": "amphibian", "treefrog": "amphibian", "caecilian": "amphibian", "bullfrog": "amphibian",
    "ant": "insect", "bee": "insect", "beetle": "insect", "butterfly": "insect",
    "cricket": "insect", "moth": "insect", "ladybug": "insect", "wasp": "insect",
    "dragonfly": "insect", "grasshopper": "insect", "ant": "insect", "termite": "insect",
    "mosquito": "insect", "fly": "insect", "flea": "insect", "caterpillar": "insect",
    "crab": "crustacean", "lobster": "crustacean", "shrimp": "crustacean", "prawn": "crustacean",
    "crayfish": "crustacean", "krill": "crustacean", "barnacle": "crustacean", "crab": "crustacean",
    "snail": "mollusk", "octopus": "mollusk", "squid": "mollusk", "clam": "mollusk",
    "oyster": "mollusk", "mussel": "mollusk", "slug": "mollusk", "cuttlefish": "mollusk",
    "scallop": "mollusk", "abalone": "mollusk", "whelk": "mollusk", "conch": "mollusk",
}

_PROP_VALUES = {
    "shift": ["morning", "afternoon", "evening", "night", "weekend", "holiday", "graveyard", "splits"],
    "team": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"],
    "language": ["Old Norse", "Umbric", "Klavish", "Doric", "Vestan", "Kaldish", "Trin", "Veyra"],
    "city": ["Port Vega", "Silverhollow", "Harborline", "Steepgate", "Wintersend", "Basalt Peak", "Dunmere", "Ashcombe"],
    "department": ["archive", "despatch", "audit", "repairs", "shipping", "returns", "cataloguing", "custody"],
    "plant": ["cedar", "maple", "birch", "spruce", "pine", "oak", "willow", "ash"],
    "instrument": ["violin", "flute", "drum", "trumpet", "harp", "cello", "clarinet", "piano"],
    "rank": ["private", "corporal", "sergeant", "lieutenant", "captain", "major", "colonel", "general"],
}

_PROP_VERBS = {
    "shift": "{n} works the {v} shift.",
    "team": "{n} is on team {v}.",
    "language": "{n} speaks {v}.",
    "city": "{n} lives in {v}.",
    "department": "{n} is assigned to the {v} department.",
    "plant": "{n} tends the {v}.",
    "instrument": "{n} plays the {v}.",
    "rank": "{n} holds the rank of {v}.",
}

_PROP_Q = {
    "shift": "Who works the {v} shift?",
    "team": "Who is on team {v}?",
    "language": "Who speaks {v}?",
    "city": "Who lives in {v}?",
    "department": "Who is assigned to the {v} department?",
    "plant": "Who tends the {v}?",
    "instrument": "Who plays the {v}?",
    "rank": "Who holds the rank of {v}?",
}

_NEG_CONTAINERS = {
    "pantry": ["salt", "rice", "oats", "honey", "beans", "lentils", "flour", "tea", "oil", "sugar", "cocoa", "cereal", "pecans", "couscous"],
    "workshop box": ["nail", "bolt", "washer", "screw", "rivet", "spring", "clamp", "nut", "hinge", "stud", "bracket", "plug", "sleeve", "pin"],
    "medicine cabinet": ["bandage", "plaster", "painkiller", "thermometer", "tape", "gauze", "syringe", "antiseptic", "tablets", "dropper", "ointment", "cloth"],
    "garden shed": ["spade", "rake", "hoe", "trowel", "shears", "fork", "hose", "gloves", "netting", "crate", "trowel", "seed tray", "stakes", "twine"],
    "toolbox": ["chisel", "vise", "mallet", "plane", "rasp", "screwdriver", "bit", "file", "plier", "rule", "gauge", "marker"],
    "kitchen drawer": ["ladle", "whisk", "tongs", "peeler", "grater", "masher", "colander", "strainer", "scoop", "roller", "brush", "timer"],
    "office cabinet": ["stapler", "punch", "tape", "marker", "stamp", "glue", "pins", "clips", "rubber band", "labels", "folder", "index cards"],
    "linen closet": ["towel", "sheet", "blanket", "pillowcase", "quilt", "duvet", "throw", "runner", "shams", "coverlet", "fitted sheet", "bathmat"],
    "pantry tray": ["cinnamon", "nutmeg", "clove", "ginger", "paprika", "cumin", "oregano", "thyme", "allspice", "cardamom", "turmeric", "saffron"],
    "lumber yard": ["oak", "maple", "birch", "cedar", "pine", "spruce", "fir", "walnut", "cherry", "ash", "beech", "hemlock"],
    "hardware bin": ["screw", "nail", "washer", "dowel", "plug", "bracket", "latch", "hook", "catch", "hinge", "bolt", "pin"],
    "sewing box": ["thread", "needle", "thimble", "scissors", "zipper", "button", "pins", "elastic", "pattern", "tape measure", "chalk", "snaps"],
}

_NEG_PHRASES = [
    "The {main} contains {items}.",
    "Inside the {main}, you will find: {items}.",
    "The following items are kept in the {main}: {items}.",
    "Everything in the {main} is listed in the inventory: {items}.",
    "The {main} holds {items}.",
    "According to the list, the {main} contains {items}.",
    "The shelf tag on the {main} reads {items}.",
    "A quick count of the {main} gives {items}.",
]

_COND_RULES = [
    {
        "text": "Orders placed before {t} are shipped the same day, while orders placed at or after {t} go out the following day.",
        "ask": "A customer placed an order at {t}. When is it shipped?",
        "tpool": [10, 11, 13, 14, 16, 17, 18],
        "threshold": 15, "hour": True,
        "below": "the same day", "above": "the following day",
        "outcomes": ["the same day", "the following day", "within a week", "after the weekend", "two days later", "never", "on request", "as soon as stock allows"],
    },
    {
        "text": "If the afternoon temperature stays below 25 degrees, the sprinklers stay off; otherwise they run.",
        "ask": "The temperature reached {t} degrees. Do the sprinklers run?",
        "tpool": [18, 19, 21, 23, 26, 28, 30, 32],
        "threshold": 25, "hour": False,
        "below": "they stay off", "above": "they run",
        "outcomes": ["they run", "they stay off", "they run once an hour", "they run continuously", "they stay on", "they run on a timer", "only at night", "only in the morning"],
    },
    {
        "text": "Members with at least 400 points qualify for the gold tier; everyone below 400 points stays in the silver tier.",
        "ask": "A member has {t} points. Which tier are they in?",
        "tpool": [120, 220, 350, 399, 410, 560, 780, 940],
        "threshold": 400, "hour": False,
        "below": "silver tier", "above": "gold tier",
        "outcomes": ["gold tier", "silver tier", "bronze tier", "platinum tier", "no tier", "applicant tier", "guest tier", "honorary tier"],
    },
    {
        "text": "Visitors 16 and older may enter the archive alone; anyone younger than 16 needs an escort.",
        "ask": "A visitor is {t} years old. May they enter alone?",
        "tpool": [7, 10, 12, 15, 17, 20, 24, 31],
        "threshold": 16, "hour": False,
        "below": "no", "above": "yes",
        "outcomes": ["yes", "no", "only on weekdays", "only during opening hours", "with a special pass", "with a member", "after registering", "never"],
    },
    {
        "text": "Trainees who score {t} or higher are boarded on the senior crew; scores below {t} keep them with the juniors.",
        "ask": "A trainee scored {t}. Which crew are they on?",
        "tpool": [41, 52, 63, 74, 86, 90, 95, 99],
        "threshold": 80, "hour": False,
        "below": "the junior crew", "above": "the senior crew",
        "outcomes": ["the senior crew", "the junior crew", "the relief crew", "the night crew", "the cadet pool", "the standby list", "the reserves", "the training squad"],
    },
    {
        "text": "Devices with at least {t} GB of free space install the full package; devices with less than {t} GB install the light package.",
        "ask": "A device has {t} GB free. Which package installs?",
        "tpool": [2, 5, 9, 12, 19, 24, 31, 42],
        "threshold": 10, "hour": False,
        "below": "the light package", "above": "the full package",
        "outcomes": ["the full package", "the light package", "the media package", "the cloud only offering", "nothing", "a partial download", "the legacy build", "the demo"],
    },
    {
        "text": "Parcels weighing under {t} kg go by air; parcels weighing {t} kg or more go by road.",
        "ask": "A parcel weighs {t} kg. How does it travel?",
        "tpool": [1, 3, 4, 7, 9, 12, 14, 18],
        "threshold": 8, "hour": False,
        "below": "by air", "above": "by road",
        "outcomes": ["by air", "by road", "by sea", "by rail", "by courier", "it does not ship", "in parts", "with a surcharge"],
    },
    {
        "text": "Rides at or after {t} are charged a night supplement; rides before {t} are at the standard fare.",
        "ask": "A ride starts at {t}. Which fare applies?",
        "tpool": [16, 18, 19, 21, 22, 23, 0, 2],
        "threshold": 22, "hour": True,
        "below": "standard fare", "above": "night supplement",
        "outcomes": ["standard fare", "night supplement", "a flat fee", "a meter surge", "a weekend rate", "a discount fare", "a waiting charge", "no fare is charged"],
    },
    {
        "text": "Batches with more than {t} units are split; batches of {t} units or fewer ship whole.",
        "ask": "A batch contains {t} units. How does it ship?",
        "tpool": [3, 6, 9, 11, 15, 20, 27, 33],
        "threshold": 12, "hour": False,
        "below": "whole", "above": "split",
        "outcomes": ["whole", "split", "at the end of the week", "in crates", "to the depot", "as single items", "in two trips", "undamaged"],
    },
    {
        "text": "Accounts averaging under {t} page views a month are archived; accounts with {t} or more stay live.",
        "ask": "An account averages {t} page views a month. Is it archived?",
        "tpool": [8, 14, 19, 24, 31, 39, 47, 55],
        "threshold": 35, "hour": False,
        "below": "archived", "above": "stays live",
        "outcomes": ["stays live", "archived", "flagged", "upgraded", "pending review", "under audit", "limited", "migrated"],
    },
    {
        "text": "Foods {t} days old or less are sold fresh; anything older than {t} days goes to the discount crate.",
        "ask": "This item is {t} days old. Where does it go?",
        "tpool": [2, 4, 6, 8, 9, 11, 13, 15],
        "threshold": 7, "hour": False,
        "below": "the fresh shelf", "above": "the discount crate",
        "outcomes": ["the fresh shelf", "the discount crate", "the freezer", "the donation bin", "the compost", "the returns pile", "the deli counter", "rejected"],
    },
    {
        "text": "Tickets bought at least {t} days in advance are at early-bird prices; tickets bought within {t} days are at the counter price.",
        "ask": "A ticket was bought {t} days in advance. What price applies?",
        "tpool": [1, 2, 4, 6, 9, 12, 14, 20],
        "threshold": 7, "hour": False,
        "below": "the counter price", "above": "the early bird price",
        "outcomes": ["the early bird price", "the counter price", "a group discount", "the student rate", "a souvenir pack", "a balcony upgrade", "a season pass", "not available"],
    },
]

_COND_OUTCOMES = [
    ["the same day", "the following day", "within a week", "after the weekend", "two days later", "never", "on request", "as soon as stock allows"],
    ["they run", "they stay off", "they run once an hour", "they run continuously", "they stay on", "they are set on a timer", "only at night", "only in the morning"],
    ["gold tier", "silver tier", "bronze tier", "platinum tier", "no tier", "applicant tier", "guest tier", "honorary tier"],
    ["yes", "no", "only on weekdays", "only during opening hours", "with a special pass", "with a member", "after registering", "never"],
]

# ---------------------------------------------------------------------------
# New-family shared vocab pools
# ---------------------------------------------------------------------------
_ITEMS = ["jaket", "lamp", "book", "hat", "mat", "stool", "rack", "bowl", "kettle",
          "vase", "clock", "mirror", "chair", "bench", "shelf", "tray", "rug", "lantern",
          "box", "pitcher", "umbrella", "toaster", "board", "crate", "jug", "basket"]
_ITEMS = ["jacket", "lamp", "book", "hat", "mat", "stool", "rack", "bowl", "kettle",
          "vase", "clock", "mirror", "chair", "bench", "shelf", "tray", "rug", "lantern",
          "box", "pitcher", "umbrella", "toaster", "board", "crate", "jug", "basket"]

_PET_LIST = ["cat", "dog", "rabbit", "parrot", "hamster", "goldfish", "turtle", "ferret",
             "guinea pig", "gerbil", "chick", "kitten"]
_DRINK_LIST = ["tea", "coffee", "water", "juice", "milk", "lemonade", "broth", "soda",
               "cider", "smoothie", "cocoa", "mocha"]
_LEAF_LIST = ["oak", "elm", "ash", "birch", "maple", "linden", "beech", "poplar",
              "willow", "chestnut", "alder", "hazel"]

_WEEKS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]

_BABY = {
    "dog": "puppy", "cat": "kitten", "horse": "foal", "cow": "calf", "sheep": "lamb",
    "pig": "piglet", "duck": "duckling", "goose": "gosling", "deer": "fawn", "goat": "kid",
    "owl": "owlet", "swan": "cygnet", "fox": "kit", "rabbit": "kitten", "frog": "tadpole",
    "chicken": "chick", "turkey": "poult", "seal": "pup", "whale": "calf", "elephant": "calf",
    "bee": "larva", "butterfly": "caterpillar", "shark": "pup", "kangaroo": "joey",
}

_MALE_FEMALE = {
    "bull": "cow", "ram": "ewe", "stallion": "mare", "rooster": "hen", "boar": "sow",
    "buck": "doe", "tom": "queen", "gander": "goose", "drake": "duck", "colt": "filly",
    "billy": "nanny", "dog": "bitch", "lion": "lioness", "tiger": "tigress", "fox": "vixen",
    "peacock": "peahen", "drone": "worker", "prince": "princess", "king": "queen", "actor": "actress",
}

_GLOSS_VERBS = ["compresses", "filters", "amplifies", "coats", "clamps", "mixes",
                "seals", "splices", "calibrates", "insulates", "stabilizes", "rotates"]
_GLOSS_OBJS = ["clay", "braided cable", "raw film", "gas feed", "spring board", "signal line",
               "plasma cell", "coolant loop", "lens blank", "draft outlet", "steam valve", "gauge probe"]
_GLOSS_SUFFIXES = ["a workshop instrument", "a laboratory fitting", "a field tool", "an installation part",
                   "a stage accessory", "a portable device", "a bench attachment", "a station component"]

_KNOWLEDGE_FACTS = [
    ("The capital of France is", "Paris"), ("The capital of Germany is", "Berlin"),
    ("The capital of Japan is", "Tokyo"), ("The capital of Australia is", "Canberra"),
    ("The capital of Brazil is", "Brasilia"), ("The capital of Canada is", "Ottawa"),
    ("The capital of India is", "New Delhi"), ("The capital of Egypt is", "Cairo"),
    ("The capital of Norway is", "Oslo"), ("The capital of Kenya is", "Nairobi"),
    ("The capital of Colombia is", "Bogota"), ("The capital of Thailand is", "Bangkok"),
    ("The currency of Japan is", "the yen"), ("The currency of Britain is", "the pound"),
    ("The currency of the United States is", "the dollar"), ("The currency of the eurozone is", "the euro"),
    ("The currency of Switzerland is", "the franc"), ("The currency of India is", "the rupee"),
    ("The currency of Brazil is", "the real"), ("The currency of Turkey is", "the lira"),
    ("The currency of Mexico is", "the peso"), ("The currency of Sweden is", "the krona"),
    ("The chemical symbol for gold is", "Au"), ("The chemical symbol for silver is", "Ag"),
    ("The chemical symbol for iron is", "Fe"), ("The chemical symbol for oxygen is", "O"),
    ("The chemical symbol for carbon is", "C"), ("The chemical symbol for copper is", "Cu"),
    ("The chemical symbol for sodium is", "Na"), ("The chemical symbol for helium is", "He"),
    ("Water is composed of two parts hydrogen and one part", "oxygen"),
    ("The largest planet in the solar system is", "Jupiter"),
    ("The planet known as the red planet is", "Mars"),
    ("The largest ocean is", "the Pacific"), ("The driest continent is", "Antarctica"),
    ("The longest river in Africa is", "the Nile"), ("The Sahara is a", "desert"),
    ("The instrument with 88 keys is the", "piano"), ("The violin has four strings made of", "gut or steel"),
    ("A triangle has this many sides", "three"), ("A hexagon has this many sides", "six"),
    ("A decade is this many years", "ten"), ("A century is this many years", "one hundred"),
    ("The color obtained by mixing red and blue is", "purple"),
    ("The color obtained by mixing blue and yellow is", "green"),
    ("Snow is made of frozen", "water"), ("Steam is water in the form of", "gas"),
    ("Humans have this many teeth in an adult set", "thirty two"), ("The human body has this many bones in an adult", "two hundred six"),
    ("The unit of electric current is the", "ampere"), ("The unit of resistance is the", "ohm"),
    ("Light travels fastest in", "a vacuum"), ("Sound does not travel in", "a vacuum"),
    ("The freezing point of water in Celsius is", "zero"), ("The boiling point of water in Celsius is", "one hundred"),
    ("A shape with four equal sides and right angles is a", "square"),
    ("A shape with three sides is a", "triangle"), ("A shape with five sides is a", "pentagon"),
    ("7 + 8 equals", "fifteen"), ("9 * 9 equals", "eighty one"),
    ("12 * 12 equals", "one hundred forty four"), ("11 * 11 equals", "one hundred twenty one"),
    ("The first month of the year is", "January"), ("The last month of the year is", "December"),
    ("The month with the fewest days is", "February"), ("A leap day is added to the month of", "February"),
    ("Horror film character inspired by Mary Shelley is", "Frankenstein"),
    ("The author of the detective who lives at 221B Baker Street is", "Arthur Conan Doyle"),
    ("The fairy tale involving a beanstalk and a giant is", "Jack and the Beanstalk"),
    ("The planet that is a gas giant with a famous ring system is", "Saturn"),
]

_SYNONYMS = [
    ("happy", "joyful"), ("sad", "unhappy"), ("quick", "fast"), ("angry", "furious"),
    ("large", "big"), ("small", "tiny"), ("smart", "clever"), ("brave", "courageous"),
    ("tired", "weary"), ("loud", "noisy"), ("quiet", "silent"), ("cold", "chilly"),
    ("hot", "warm"), ("wet", "damp"), ("dry", "arid"), ("tough", "hardy"),
    ("frail", "weak"), ("bright", "radiant"), ("dull", "boring"), ("sharp", "keen"),
    ("smooth", "even"), ("rough", "coarse"), ("free", "free"), ("grand", "magnificent"),
    ("funny", "humorous"), ("gloomy", "somber"), ("rich", "wealthy"), ("poor", "destitute"),
    ("young", "youthful"), ("old", "aged"), ("slim", "slender"), ("thick", "dense"),
    ("firm", "solid"), ("lazy", "sluggish"), ("eager", "keen"), ("nervous", "anxious"),
    ("sure", "certain"), ("skilled", "expert"), ("famous", "renowned"), ("ordinary", "common"),
    ("strange", "odd"), ("unusual", "rare"), ("gigantic", "enormous"), ("minuscule", "tiny"),
    ("accurate", "precise"), ("genuine", "authentic"), ("fragile", "delicate"), ("robust", "sturdy"),
]

_ANTONYMS = [
    ("happy", "sad"), ("hot", "cold"), ("big", "small"), ("fast", "slow"),
    ("loud", "quiet"), ("bright", "dark"), ("tall", "short"), ("rich", "poor"),
    ("strong", "weak"), ("kind", "cruel"), ("clean", "dirty"), ("early", "late"),
    ("start", "end"), ("open", "close"), ("push", "pull"), ("win", "lose"),
    ("day", "night"), ("wet", "dry"), ("sharp", "blunt"), ("thick", "thin"),
    ("young", "old"), ("full", "empty"), ("heavy", "light"), ("rough", "smooth"),
    ("brave", "cowardly"), ("smart", "foolish"), ("polite", "rude"), ("cheap", "expensive"),
    ("easy", "hard"), ("sweet", "sour"), ("wide", "narrow"), ("soft", "hard"),
    ("deep", "shallow"), ("far", "near"), ("up", "down"), ("left", "right"),
    ("summer", "winter"), ("morning", "evening"), ("victory", "defeat"), ("order", "chaos"),
]

_REL_TRIPLES = list(_BABY.items()) + list(_MALE_FEMALE.items())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _row(context, question, options, correct, family, difficulty, generated_by, probs=None):
    if probs is None:
        probs = [0.05 / (N_OPTIONS - 1)] * N_OPTIONS
        if correct is not None:
            probs[correct] = 0.95
    return {
        "id": str(uuid.uuid4()),
        "task": family,
        "difficulty": difficulty,
        "context": context,
        "question": question,
        "options": options,
        "correct_index": correct,
        "teacher_probs": probs,
        "label_source": {"model": "synthetic-templates", "temperature": 0.0, "n_samples": 1, "vote": "template"},
        "meta": {"family": family, "generated_by": generated_by},
    }


def _opt_with(rng, answer, distractors, k=7):
    opts = [answer]
    ds = [str(d) for d in distractors if str(d) != str(answer)]
    rng.shuffle(ds)
    opts.extend(ds[: k])
    while len(opts) < N_OPTIONS:
        opts.append(f"placeholder {rng.randrange(1000)}")
    rng.shuffle(opts)
    return opts, opts.index(answer)


def _fill_options(rng, answer, base, min_v, max_v):
    opts = {str(answer)}
    opts.update(str(x) for x in base if min_v <= x <= max_v)
    while len(opts) < N_OPTIONS:
        opts.add(str(rng.randint(max(1, min_v), max_v)))
    options = list(opts)
    rng.shuffle(options)
    return options, options.index(str(answer))


def _fmt_hour(h):
    if h == 0:
        return "12 am"
    if h == 12:
        return "12 pm"
    return f"{h} am" if h < 12 else f"{h - 12} pm"


# ---------------------------------------------------------------------------
# existing families
# ---------------------------------------------------------------------------
def gen_retrieval(rng):
    obj, attrs = rng.choice(list(_FACTS.items()))
    attr = rng.choice(list(attrs))
    value = attrs[attr]
    phrase = rng.choice(_RETRIEVAL_PHRASES).format(obj=obj, attr=attr, value=value)
    pool = list(_FACTS.items())
    rng.shuffle(pool)
    options = []
    for o, a in pool:
        if len(options) >= N_OPTIONS - 1:
            break
        if a[attr] != value:
            options.append(a[attr])
    while len(options) < N_OPTIONS - 1:
        options.append("an unknown value")
    rng.shuffle(options)
    pos = rng.randrange(len(options) + 1)
    options.insert(pos, value)
    q = rng.choice([
        f"What is the {attr} of {obj}?",
        f"What {attr} does {obj} have?",
        f"According to the source, what is the {attr} of {obj}?",
        f"Find the {attr} listed for {obj}.",
    ])
    return _row(phrase, q, options, pos, "retrieval", "easy", "template:attr-value")


def gen_paraphrase(rng):
    intent = rng.choice(list(_INTENT))
    phrase = rng.choice(_INTENT[intent])
    context = rng.choice([
        f"The customer {phrase}.",
        f"A caller {phrase}.",
        f"We heard that the customer {phrase}.",
    ])
    options = [_INTENT_LABEL[k] for k in _INTENT]
    rng.shuffle(options)
    correct = options.index(_INTENT_LABEL[intent])
    q = rng.choice([
        "What does the customer want?", "What action is the customer asking for?",
        "What should the support agent do?", "What is the customer requesting?",
    ])
    return _row(context, q, options, correct, "paraphrase", "medium", "template:intent-paraphrase")


def gen_order(rng):
    events = rng.sample(_EVENTS, 3)
    a, b, c = events
    context = rng.choice(_ORDER_PHRASES).format(a=a, b=b, c=c)
    pos = rng.randrange(3)
    answer = [a, b, c][pos]
    label = ["first", "second", "third"][pos]
    options = rng.sample([e for e in _EVENTS], N_OPTIONS)
    if answer not in options:
        options = options[: N_OPTIONS - 1] + [answer]
    rng.shuffle(options)
    return _row(context, f"Which event happened {label}?", options, options.index(answer), "order", "hard", "template:sequence")


def gen_ambiguity(rng):
    context = rng.choice(_AMBIGUITY_PH)
    options = rng.sample(_AMB_OPTIONS, len(_AMB_OPTIONS))
    probs = list(_AMB_PROBS)
    rng.shuffle(probs)
    return _row(context, "What most likely caused the interrupt?", options, None, "ambiguity", "hard", "template:underdetermined", probs=probs)


def gen_permuted(rng):
    row = gen_retrieval(rng)
    options = list(row["options"])
    correct = row["correct_index"]
    answer = options.pop(correct)
    rng.shuffle(options)
    pos = rng.randrange(len(options) + 1)
    options.insert(pos, answer)
    return _row(row["context"], row["question"], options, pos, "permuted", "easy", "permuted:template:attr-value")


def gen_numeric(rng):
    who = rng.choice(_NAMES)
    i1, i2 = rng.sample(_ITEMS, 2)
    c1, c2 = rng.randint(3, 60), rng.randint(3, 60)
    phrase = rng.choice([
        f"{who} bought a {i1} for {c1} euros and a {i2} for {c2} euros.",
        f"{who} paid {c1} euros for a {i1} and {c2} euros for a {i2}.",
        f"The receipt shows {c1} euros for the {i1} and {c2} euros for the {i2}, both bought by {who}.",
        f"{who} picked up a {i1} at {c1} euros and a {i2} at {c2} euros during the sale.",
    ])
    qtype = rng.choice(("total", "more", "difference"))
    if qtype == "total":
        ans, q = c1 + c2, f"How much did {who} spend in total?"
    elif qtype == "more":
        hi_i, hi_c = (i1, c1) if c1 >= c2 else (i2, c2)
        ans, q = hi_c, f"What did the {hi_i} cost?"
    else:
        hi_i, hi_c = (i1, c1) if c1 >= c2 else (i2, c2)
        lo_i, lo_c = (i2, c2) if hi_i == i1 else (i1, c1)
        ans, q = hi_c - lo_c, f"How much more did the {hi_i} cost than the {lo_i}?"
    options, correct = _fill_options(rng, ans, [c1, c2, c1 + c2, abs(c1 - c2)], 1, 140)
    return _row(phrase, q, options, correct, "numeric", "easy", "template:numeric-arithmetic")


def gen_spatial(rng):
    place, objs = rng.choice(list(_SPATIAL_PLACES.items()))
    row = rng.sample(sorted(set(objs)), 5)
    context = rng.choice([
        f"On the {place}, from left to right, sit: {', '.join(row)}.",
        f"Laid out left to right on the {place} are {', '.join(row)}.",
        f"On the {place}, {row[0]} comes first, then {row[1]}, then {row[2]}, then {row[3]}, and {row[4]} is furthest right.",
        f"Reading across the {place}: {', '.join(row)}.",
    ])
    kind = rng.choice(("right", "left", "rightmost", "leftmost", "middle"))
    if kind == "right":
        i = rng.randrange(4)
        ans, q = row[i + 1], f"Which object is immediately to the right of the {row[i]}?"
    elif kind == "left":
        i = rng.randrange(1, 5)
        ans, q = row[i - 1], f"Which object is immediately to the left of the {row[i]}?"
    elif kind == "rightmost":
        ans, q = row[4], "Which object is furthest to the right?"
    elif kind == "leftmost":
        ans, q = row[0], "Which object is furthest to the left?"
    else:
        ans, q = row[2], "Which object is in the middle of the arrangement?"
    distractors = [o for o in sorted(set(objs)) if o not in row]
    options = list(row) + rng.sample(distractors, N_OPTIONS - len(row))
    rng.shuffle(options)
    return _row(context, q, options, options.index(ans), "spatial", "easy", "template:spatial-row")


def gen_compare(rng):
    names = rng.sample(_NAMES, 8)
    attr = rng.choice(_COMPARE_ATTRS)
    unit, verb = attr["unit"], attr["verb"]
    ranges = {"cm": (120, 200), "kg": (30, 120), "km/h": (10, 130),
              "eur": (5, 500), "years": (5, 90), "points": (10, 300),
              "pages": (80, 900)}
    lo_v, hi_v = ranges[unit]
    values = rng.sample(range(lo_v, hi_v), 8)
    fmt = [f"{n} is {v} {unit} {verb}." for n, v in zip(names, values)]
    context = " ".join(fmt)
    hi, low = attr["q"]
    want_hi = rng.choice((True, False))
    ans = names[values.index(max(values) if want_hi else min(values))]
    q = f"Who has the {hi if want_hi else low}?"
    options = list(names)
    rng.shuffle(options)
    return _row(context, q, options, options.index(ans), "compare", "easy", "template:comparison")


def gen_oddoneout(rng):
    cats = list(_ODD_CATS)
    main, other = rng.sample(cats, 2)
    pool_other = [o for o in _ODD_CATS[other] if o not in _ODD_CATS[main]]
    items = rng.sample(sorted(set(_ODD_CATS[main])), 7)
    intruder = rng.choice(pool_other)
    context = rng.choice([
        f"The {main} include: {', '.join(items)}.",
        f"In the box labelled {main} are {', '.join(items)}.",
        f"Someone put these items in the {main} basket: {', '.join(items)}.",
        f"The {main} collection consists of {', '.join(items)}.",
    ])
    options = items + [intruder]
    rng.shuffle(options)
    return _row(context, "Which item does not belong?", options, options.index(intruder), "oddoneout", "easy", "template:oddoneout")


def gen_category(rng):
    inst = rng.choice(sorted(set(_CAT_MAP)))
    cls = _CAT_MAP[inst]
    context = rng.choice([
        f"The {inst} belongs to the {cls} group.",
        f"Biologists place the {inst} among the {cls}s.",
        f"The {inst} is a member of the {cls} group.",
        f"Classified many times over, the {inst} is a {cls}.",
    ])
    options = list(_CAT_CLASSES)
    rng.shuffle(options)
    q = f"To which group does the {inst} belong?"
    return _row(context, q, options, options.index(cls), "category", "easy", "template:instance-class")


def gen_property(rng):
    attr, values = rng.choice(list(_PROP_VALUES.items()))
    verb = _PROP_VERBS[attr]
    names = rng.sample(_NAMES, 8)
    target = rng.choice(names)
    tv = rng.choice(values)
    assigned = {target: tv}
    others = [n for n in names if n != target]
    for n in others:
        assigned[n] = rng.choice([v for v in values if v != tv])
    context = ". ".join(verb.format(n=n, v=assigned[n]) for n in names) + "."
    options = list(names)
    rng.shuffle(options)
    q = _PROP_Q[attr].format(v=tv)
    return _row(context, q, options, options.index(target), "property", "easy", "template:attribute")


def gen_negation(rng):
    conts = list(_NEG_CONTAINERS)
    main, other = rng.sample(conts, 2)
    items = rng.sample(sorted(set(_NEG_CONTAINERS[main])), 7)
    intruder = rng.choice([x for x in sorted(set(_NEG_CONTAINERS[other])) if x not in _NEG_CONTAINERS[main]])
    context = rng.choice(_NEG_PHRASES).format(main=main, items=", ".join(items))
    options = items + [intruder]
    rng.shuffle(options)
    return _row(context, f"Which of these is NOT in the {main}?", options, options.index(intruder), "negation", "easy", "template:set-negation")


def gen_conditional(rng):
    rule = rng.choice(_COND_RULES)
    t = rng.choice(rule["tpool"])
    if rule["hour"]:
        td = _fmt_hour(t)
        text = rule["text"].format(t=td)
        ask = rule["ask"].format(t=td)
    else:
        text = rule["text"].format(t=t)
        ask = rule["ask"].format(t=t)
    ans = rule["below"] if t < rule["threshold"] else rule["above"]
    options = list(rule["outcomes"])
    rng.shuffle(options)
    return _row(text, ask, options, options.index(ans), "conditional", "easy", "template:conditional-rule")


# ---------------------------------------------------------------------------
# new families
# ---------------------------------------------------------------------------
def gen_arithmetic(rng):
    who = rng.choice(_NAMES)
    items = rng.sample(_ITEMS, 2)
    c1, c2 = rng.randint(4, 80), rng.randint(4, 80)
    ctx_buy = f"{who} bought {items[0]} for {c1} euros and {items[1]} for {c2} euros."
    qtype = rng.choice(("sum", "difference", "product", "average", "percent"))
    if qtype == "sum":
        ans = c1 + c2
        q = "What is the combined cost of the two purchases?"
        base = [c1, c2, abs(c1 - c2)]
    elif qtype == "difference":
        ans = abs(c1 - c2)
        q = "How much more did one purchase cost than the other?"
        base = [c1, c2, c1 + c2]
    elif qtype == "product":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        ans = a * b
        ctx_buy = f"A tray holds {a} rows of {b} tiles."
        q = "How many tiles are on the tray?"
        base = [a, b, a + b]
    elif qtype == "average":
        a, b, cc = rng.randint(10, 60), rng.randint(10, 60), rng.randint(10, 60)
        ans = (a + b + cc) // 3
        ctx_buy = f"Three readings came in at {a}, {b} and {cc}."
        q = "What is their average (rounded down)?"
        base = [a, b, cc, a + b + cc]
    else:
        tot = rng.randint(20, 200)
        pct = rng.choice([10, 20, 25, 50])
        ans = tot * pct // 100
        ctx_buy = f"A bill of {tot} euros is discounted by {pct} percent."
        q = "How many euros are taken off?"
        base = [tot, tot * (pct + 10) // 100, tot * (pct - 5) // 100]
    lo, hi = 1, max(400, ans * 3)
    options, correct = _fill_options(rng, ans, base, 1, hi)
    return _row(ctx_buy, q, options, correct, "arithmetic", "medium", "template:arithmetic")


def gen_counting(rng):
    conts = rng.sample(sorted(set(_NEG_CONTAINERS)), 2)
    c1, c2 = conts
    items1 = rng.sample(sorted(set(_NEG_CONTAINERS[c1])), rng.randint(3, 5))
    items2 = rng.sample(sorted(set(_NEG_CONTAINERS[c2])), rng.randint(3, 5))
    context = f"In the {c1} we see {', '.join(items1)}. In the {c2} we see {', '.join(items2)}."
    qtype = rng.choice(("c1", "c2", "both", "total"))
    if qtype == "c1":
        ans = len(items1)
        q = f"How many items are in the {c1}?"
    elif qtype == "c2":
        ans = len(items2)
        q = f"How many items are in the {c2}?"
    elif qtype == "both":
        shared = set(items1) & set(items2)
        ans = len(shared)
        q = f"How many items appear in both places?"
        context = f"In the {c1} we see {', '.join(items1)}. In the {c2} we see {', '.join(items2)}."
    else:
        ans = len(items1) + len(items2)
        q = "How many items are listed in total?"
    options, correct = _fill_options(rng, ans, [len(items1), len(items2), len(items1) + len(items2)], 1, 12)
    return _row(context, q, options, correct, "counting", "medium", "template:counting")


def gen_sets(rng):
    cats = rng.sample(sorted(set(_ODD_CATS)), 2)
    a, b = cats
    poolA = sorted(set(_ODD_CATS[a]))
    poolB = sorted(set(_ODD_CATS[b]))
    A = set(rng.sample(poolA, 4))
    B = set(rng.sample(poolB, 4))
    inter = A & B
    context = f"Group A is {', '.join(sorted(A))}. Group B is {', '.join(sorted(B))}."
    kind = rng.choice(("only_a", "only_b", "either", "both"))
    if kind == "only_a":
        ans = rng.choice(sorted(A - B - inter))
        q = f"Which of these is in Group A but not in Group B?"
        opts = sorted((A - B) | (B - A)) + sorted(inter)
    elif kind == "only_b":
        ans = rng.choice(sorted(B - inter))
        q = "Which of these is in Group B but not in Group A?"
        opts = sorted((A - B) | (B - A)) + sorted(inter)
    elif kind == "either":
        ans = rng.choice(sorted((A - B) | (B - A) | inter))
        q = "Which of these appears in at least one group?"
        opts = sorted((A | B))
    else:
        ans = rng.choice(sorted(inter)) if inter else None
        q = "Which of these appears in both groups at once?"
        opts = sorted(A | B)
        if ans is None:
            return gen_sets(rng)
    rng.shuffle(opts)
    options, correct = _opt_with(rng, ans, opts, N_OPTIONS - 1)
    return _row(context, q, options, correct, "sets", "medium", "template:set-ops")


def gen_beforeafter(rng):
    events = rng.sample(_EVENTS, 3)
    a, b, c = events
    context = rng.choice([
        f"{a} took place before {b}, and {c} came after {b}.",
        f"The sequence ran {a}, then {b}, then {c}.",
        f"First came {a}; {b} followed; {c} wrapped things up.",
    ])
    kind = rng.choice(("before_b", "before_c", "after_a", "after_b"))
    if kind == "before_b":
        ans, ref = a, b
    elif kind == "before_c":
        ans, ref = b, c
    elif kind == "after_a":
        ans, ref = b, a
    else:
        ans, ref = c, b
    q = f"Which event happened {'before' if kind.startswith('before') else 'after'} {ref}?"
    options = rng.sample([e for e in _EVENTS if e != ans], N_OPTIONS)
    options = options[: N_OPTIONS - 1] + [ans]
    rng.shuffle(options)
    return _row(context, q, options, options.index(ans), "beforeafter", "medium", "template:before-after")


def gen_ordinal(rng):
    objs = rng.sample(_ITEMS, 8)
    context = rng.choice([
        f"In the race the finishing order was: {', '.join(objs)}.",
        f"The ranking from first to last reads: {', '.join(objs)}.",
        f"Results, top to bottom: {', '.join(objs)}.",
    ])
    pos = rng.randrange(8)
    labels = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "last"]
    label = labels[pos]
    ans = objs[pos]
    options = list(objs)
    rng.shuffle(options)
    q = f"Which one finished {label}?"
    return _row(context, q, options, options.index(ans), "ordinal", "medium", "template:ordinal")


def gen_sequence_next(rng):
    kind = rng.choice(("add", "mult", "alt"))
    a, d = rng.randint(1, 9), rng.randint(2, 9)
    if kind == "add":
        seq = [a + d * i for i in range(5)]
        ans = a + d * 5
        dist = [a + d * (5 + rng.choice([1, 2])), a + d * (3), a * 2 + d * 5]
    elif kind == "mult":
        a = rng.randint(1, 4)
        m = rng.randint(2, 4)
        seq = [a * (m ** i) for i in range(5)]
        ans = a * (m ** 5)
        dist = [a * (m ** 4) + 1, a * ((m + 1) ** 4), a * (m ** 3) * 2]
    else:
        p = rng.randint(5, 20)
        q2 = rng.randint(1, 5)
        seq = [p + (q2 if i % 2 == 0 else -q2) + 0 for i in range(5)]
        s = seq
        ans = p - q2 if s[-1] == p + q2 else p + q2
        dist = [s[-1], p + q2 if ans == p - q2 else p - q2, s[-1] + 1]
    seq.append(ans)
    seq2 = seq[:5]
    context = f"The sequence begins: {', '.join(map(str, seq2))}."
    opts = {str(ans)} | {str(x) for x in dist if x != ans}
    while len(opts) < N_OPTIONS:
        opts.add(str(rng.randint(1, 120)))
    options = list(opts)
    rng.shuffle(options)
    return _row(context, "What number comes next in the sequence?", options, options.index(str(ans)), "sequence_next", "hard", "template:sequence-next")


def gen_timeclock(rng):
    hh = rng.randint(0, 23)
    mm = rng.choice([0, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    add = rng.choice([25, 30, 35, 40, 45, 55, 60, 70, 90, 110, 135])
    now = hh * 60 + mm
    fut = (now + add) % 1440
    fh, fm = fut // 60, fut % 60
    ansstr = f"{fh % 12 if fh % 12 else 12}:{fm:02d}"
    ampm = " am" if fh < 12 else " pm"
    context = f"It is {_fmt_hour(hh)} and {mm} minutes past."
    q = f"After {add} minutes, what time will it be?"
    options, correct = _opt_with(rng, ansstr + ampm, [f"{(fh % 12 if fh % 12 else 12)}:{fm:02d} am", f"{fh}:{fm:02d}", _fmt_hour(hh)], 7)
    return _row(context, q, options, correct, "timeclock", "hard", "template:clock")


def gen_dates(rng):
    kind = rng.choice(("day", "month", "monthcount"))
    if kind == "day":
        dow = rng.randrange(7)
        add = rng.randint(2, 14)
        ans = _WEEKS[(dow + add) % 7]
        context = f"Today is {_WEEKS[dow]}."
        q = f"What day of the week will it be in {add} days?"
        options = list(_WEEKS)
        rng.shuffle(options)
        options.append("None of the above")
        rng.shuffle(options)
        return _row(context, q, options, options.index(ans), "dates", "medium", "template:day-math")
    if kind == "month":
        mo = rng.randrange(12)
        add = rng.randint(1, 24)
        ans = _MONTHS[(mo + add) % 12]
        context = f"This month is {_MONTHS[mo]}."
        q = f"What month will it be {add} month(s) from now?"
        options, correct = _opt_with(rng, ans, _MONTHS, N_OPTIONS - 1)
        return _row(context, q, options, correct, "dates", "medium", "template:month-math")
    mo = rng.randrange(12)
    nxt = _MONTHS[(mo + 1) % 12]
    ans = _MONTHS[(mo + 2) % 12]
    context = f"The month after {_MONTHS[mo]} is {nxt}."
    q = f"Which month comes immediately after {nxt}?"
    options, correct = _opt_with(rng, ans, _MONTHS, N_OPTIONS - 1)
    return _row(context, q, options, correct, "dates", "medium", "template:month-after")


def gen_negprop(rng):
    attr, values = rng.choice(list(_PROP_VALUES.items()))
    verb = _PROP_VERBS[attr]
    names = rng.sample(_NAMES, 8)
    tv = rng.choice(values)
    alt = rng.choice([v for v in values if v != tv])
    exclude = rng.choice(names)
    context_parts = []
    for n in names:
        v = alt if n == exclude else tv
        context_parts.append(verb.format(n=n, v=v))
    context = ". ".join(context_parts) + "."
    q = f"Who does NOT work the {tv} shift?"
    if attr == "team":
        q = f"Who is not on team {tv}?"
    elif attr == "language":
        q = f"Who does not speak {tv}?"
    elif attr == "city":
        q = f"Who does not live in {tv}?"
    elif attr == "department":
        q = f"Who is NOT assigned to the {tv} department?"
    elif attr == "plant":
        q = f"Who does not tend the {tv}?"
    elif attr == "instrument":
        q = f"Who does not play the {tv}?"
    elif attr == "rank":
        q = f"Who does not hold the rank of {tv}?"
    options = names[:]
    rng.shuffle(options)
    return _row(context, q, options, options.index(exclude), "negprop", "medium", "template:attribute-negation")


def gen_analogy_tmpl(rng):
    a, b = rng.choice(list(_BABY.items()))
    c = rng.choice(sorted(set(_BABY)))
    context = f"{a} is to {b} as {c} is to what?"
    opts = list(set(_BABY.values()))
    options, correct = _opt_with(rng, _BABY[c], sorted(set(_BABY.values())), N_OPTIONS - 1)
    return _row(context, "Complete the analogy.", options, correct, "analogy", "medium", "template:analogy-baby")


def gen_definition_tmpl(rng):
    term = f"f{''.join(rng.choice('aeiou') for _ in range(2))}g{rng.randint(10, 99)}"
    verb = rng.choice(_GLOSS_VERBS)
    obj = rng.choice(_GLOSS_OBJS)
    kind = rng.choice(_GLOSS_SUFFIXES)
    context = f"A {term} is a {kind} that {verb} the {obj} before use."
    q = f"What does the term \u201c{term}\u201d mean?"
    ans = f"a {kind} that {verb} the {obj} before use"
    opts = [ans]
    covered = set()
    for _ in range(12):
        v2 = rng.choice(_GLOSS_VERBS)
        o2 = rng.choice(_GLOSS_OBJS)
        k2 = rng.choice(_GLOSS_SUFFIXES)
        s = f"a {k2} that {v2} the {o2} before use"
        if s not in covered:
            covered.add(s)
            opts.append(s)
    options, correct = _opt_with(rng, ans, opts[1:], N_OPTIONS - 1)
    return _row(context, q, options, correct, "define", "easy", "template:gloss")


def gen_knowledge_tmpl(rng):
    ctx, ans = rng.choice(_KNOWLEDGE_FACTS)
    options, correct = _opt_with(rng, ans, [v for _, v in _KNOWLEDGE_FACTS], N_OPTIONS - 1)
    return _row(ctx, "Choose the answer that correctly completes the statement.", options, correct, "knowledge", "easy", "template:factbank")


def gen_synonym(rng):
    w, syn = rng.choice(_SYNONYMS)
    context = f"The word of the day is \u201c{w}\u201d."
    q = f"Which option is a synonym of \u201c{w}\u201d?"
    options, correct = _opt_with(rng, syn, [s for _, s in _SYNONYMS] + [b for a, b in _ANTONYMS if b != syn], N_OPTIONS - 1)
    return _row(context, q, options, correct, "synonym", "medium", "template:synonym")


def gen_antonym(rng):
    w, ant = rng.choice(_ANTONYMS)
    context = f"Consider the word \u201c{w}\u201d."
    q = f"Which option is an antonym of \u201c{w}\u201d?"
    options, correct = _opt_with(rng, ant, [b for a, b in _ANTONYMS] + [s for _, s in _SYNONYMS if s != ant], N_OPTIONS - 1)
    return _row(context, q, options, correct, "antonym", "medium", "template:antonym")


def gen_transitivity(rng):
    names = rng.sample(_NAMES, 4)
    a, b, c, d = names
    rel = rng.choice(["taller", "faster", "heavier", "wealthier", "older", "further along the trail"])
    context = rng.choice([
        f"{a} is {rel} than {b}, and {b} is {rel} than {c}, and {c} is {rel} than {d}.",
        f"Ranked by who is {rel}: {a} leads {b}, who leads {c}, who leads {d}.",
        f"{a} > {b} > {c} > {d} in terms of being {rel}.",
    ])
    kind = rng.choice(("most", "least", "second", "third"))
    if kind == "most":
        ans, q = a, f"Who is the {rel} of the four?"
    elif kind == "least":
        ans, q = d, f"Who is the least {rel} of the four?"
    elif kind == "second":
        ans, q = b, f"Who is the second-most {rel}?"
    else:
        ans, q = c, f"Who is the third-most {rel}?"
    options, correct = _opt_with(rng, ans, names, N_OPTIONS - 1)
    return _row(context, q, options, correct, "transitivity", "hard", "template:transitive")


def gen_distribution(rng):
    total = rng.choice([60, 90, 100, 120, 150, 180, 210, 240])
    r1, r2, r3 = rng.sample(range(1, 5), 3)
    who = rng.sample(_NAMES, 3)
    s = r1 + r2 + r3
    amts = [total * r1 // s, total * r2 // s, total * r3 // s]
    ctx = f"A stack of {total} tokens is shared among {who[0]}, {who[1]} and {who[2]} in the ratio {r1}:{r2}:{r3}."
    pos = rng.randrange(3)
    ans = amts[pos]
    q = f"How many tokens does {who[pos]} receive?"
    options, correct = _fill_options(rng, ans, amts, 1, total)
    return _row(ctx, q, options, correct, "distribution", "hard", "template:ratio")


GENERATORS = {
    "retrieval": gen_retrieval,
    "paraphrase": gen_paraphrase,
    "order": gen_order,
    "ambiguity": gen_ambiguity,
    "permuted": gen_permuted,
    "numeric": gen_numeric,
    "spatial": gen_spatial,
    "compare": gen_compare,
    "oddoneout": gen_oddoneout,
    "category": gen_category,
    "property": gen_property,
    "negation": gen_negation,
    "conditional": gen_conditional,
    "arithmetic": gen_arithmetic,
    "counting": gen_counting,
    "sets": gen_sets,
    "beforeafter": gen_beforeafter,
    "ordinal": gen_ordinal,
    "sequence_next": gen_sequence_next,
    "timeclock": gen_timeclock,
    "dates": gen_dates,
    "negprop": gen_negprop,
    "analogy": gen_analogy_tmpl,
    "define": gen_definition_tmpl,
    "knowledge": gen_knowledge_tmpl,
    "synonym": gen_synonym,
    "antonym": gen_antonym,
    "transitivity": gen_transitivity,
    "distribution": gen_distribution,
}

_ALL_FAMILIES = ",".join(GENERATORS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SYNTHETIC_DATA))
    ap.add_argument("--per-family", type=int, default=1000)
    ap.add_argument("--families", default=_ALL_FAMILIES)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    names = [n.strip() for n in args.families.split(",") if n.strip()]
    assert set(names) <= set(GENERATORS), f"unknown families: {set(names) - set(GENERATORS)}"
    for name in names:
        gen = GENERATORS[name]
        for _ in range(args.per_family):
            rows.append(gen(rng))

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    counts = {}
    for r in rows:
        counts[r["meta"]["family"]] = counts.get(r["meta"]["family"], 0) + 1
    for k, v in counts.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()