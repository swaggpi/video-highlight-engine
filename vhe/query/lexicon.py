"""Every word list and pattern the brief rules (vhe.query.rules) match, English and German: moods, speech intent,
durations and counts, opener/closer, the plan (shots, avoid and exclude clauses, spoken topics, live sound) and the
visual cues (framing, faces, hook, readable text, centring). Patterns run on the normalised, lower-cased brief."""
from __future__ import annotations

import re

# ---- mood, speech, lengths, counts ---------------------------------------------------------------------------------
CALM = ["chill", "calm", "relaxed", "relaxing", "peaceful", "quiet", "slow", "serene", "tranquil", "cozy", "scenic",
        "ruhig", "ruhige", "entspannt", "entspannte", "gemütlich", "friedlich", "langsam", "chillig"]
ENERGY = ["high-energy", "high energy", "energetic", "action", "action-packed", "exciting", "intense", "fast",
          "hype", "wild", "crazy", "epic", "energiegeladen", "energiegeladene", "actiongeladen", "spannend",
          "spannende", "schnell", "schnelle", "aufregend", "wilde"]
# Speech intent. Verbs carry no visual meaning and are removed from the content text; the nouns in SPEECH_VISUAL
# ("toast", "interview", "to camera") also describe what is visible and stay in it.
SPEECH_VERBS = ["speaks", "speaking", "speak", "talks", "talking", "talk", "says", "saying", "addresses",
                "addressing", "speech", "speeches", "announcement", "announcements", "spricht", "sprechen", "redet",
                "reden", "rede", "sagt"]
SPEECH_VISUAL = ["to camera", "to the camera", "interview", "interviews", "toast", "toasts", "presenter",
                 "conversation", "gespräch", "in die kamera"]
SPEECH_ON = SPEECH_VERBS + SPEECH_VISUAL
SHORT = ["short", "quick", "kurz", "kurze", "kurzen"]
LONG = ["long", "longer", "lang", "lange", "langen"]
CLIP_NOUNS = (r"(?:clips?|scenes?|shots?|moments?|highlights|parts|stretch(?:es)?|cutaways?|selects?|holds?|cuts|"
              r"hooks?|first-shots?|"
              r"szenen?|momente|stellen|aufnahmen)")
# A platform named next to a duration ("30 sek für insta") makes it the runtime of the whole post.
BUDGET_NOUNS = (r"(?:recaps?|edit|video|reels?|montage|highlights?|cut|film|trailer|teaser|sizzle|summary|supercut|"
                r"compilation|rückblick|zusammenfassung|zusammenschnitt|schnitt|insta|instagram|tiktok|youtube|piece|"
                r"package|pkg|spot|segment)")
FORMAT = ["recap", "recaps", "edit", "reel", "montage", "highlight", "highlights", "supercut", "compilation",
          "summary", "trailer", "teaser", "sizzle", "video", "b-roll", "broll", "rückblick", "zusammenfassung",
          "zusammenschnitt"]
FILLER = ["show me", "find me", "find", "give me", "i want", "i need", "make", "create", "zeig mir", "finde",
          "ich will", "ich brauche", "scenes", "scene", "clips", "clip", "shots", "shot", "moments", "moment",
          "parts", "szenen", "szene", "momente", "stellen", "aufnahmen",
          "with", "mit", "and", "und", "the", "die", "der", "das", "some", "ein paar", "einige", "absolutely",
          "definitely", "bitte", "please"]
EDGE_STOP = {"a", "an", "the", "of", "with", "for", "from", "in", "on", "at", "and", "or", "to", "some", "where",
             "when", "ein", "eine", "einer", "einem", "der", "die", "das", "den", "dem", "des", "von", "vom", "im",
             "am", "mit", "und"}
UNIT = r"(?:s|sec|secs|second|seconds|sek|sekunde|sekunden|min|mins|minute|minutes|minuten)\b"
NUM = r"\d+(?:\.\d+)?"
VALUE = rf"(?P<a>{NUM})\s*(?P<ua>{UNIT})?(?:\s*(?:-|to|bis)\s*(?P<b>{NUM}))?[\s-]*(?P<u>{UNIT})"
WORDS = r"(?:[^\W\d][\w-]*\s+)"  # one word and its trailing space; never crosses digits or punctuation
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45,
                "fifty": 50, "sixty": 60, "ninety": 90, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6,
                "sieben": 7, "acht": 8, "zehn": 10, "zwölf": 12, "fünfzehn": 15, "zwanzig": 20, "dreißig": 30,
                "vierzig": 40, "fünfzig": 50, "sechzig": 60, "neunzig": 90}
MODIFIERS = (r"(?:(?P<total>in total|total|overall|altogether|insgesamt|gesamt)|"
             r"(?P<le>under|less than|below|at most|max(?:imum)?|no (?:longer|more) than|up to|within|unter|"
             r"höchstens|maximal|bis zu)|"
             r"(?P<ge>over|more than|at least|min(?:imum)?|über|mindestens)|"
             r"(?P<about>about|around|roughly|approx(?:imately)?|circa|ca\.?|ungefähr|etwa|rund))\s*$")
# The description runs to the next punctuation, "then", or the other edge's verb phrase ("... and end with"), so
# "end on the giant flag and the chanting crowd at the end of the route" keeps its "and ..." and its "end of".
EDGE_VERBS = r"(?:end|ends|close|closes|finish|finishes|start|starts|open|opens|begin|begins|ende|starte|beginne)"
EDGE_OF = (r"\s+(?:(?:it|the\s+(?:edit|video|recap|reel|film|montage))\s+)?(?:with|on|mit|auf)\s+"
           r"(?:(?:a|an|the|some|ein|eine|einem|einer|dem|der)\s+)?(?P<x>[^,;.]+?)\s*"
           rf"(?=[,;.]|\b(?:then|dann)\b|\b(?:(?:and|und)\s+)?{EDGE_VERBS}\s+(?:it\s+|the\s+\w+\s+)?"
           r"(?:with|on|mit|auf)\b|$)")
OPENER = (r"\b(?:open|opens|opening|start|starts|starting|begin|begins|beginning|kick off|lead|beginne|beginnen|"
          r"starte|starten|fange an)" + EDGE_OF)
CLOSER = (r"\b(?:end|ends|ending|close|closes|closing|finish|finishes|finishing|wrap up|wrap it up|ende|enden|"
          r"schließe|schließen|höre auf)" + EDGE_OF)
# "every toast" is exhaustive on its own; "all"/"each" only with a generic noun ("all smiles" is not).
GENERIC = r"(?:moments?|times?|clips?|shots?|scenes?|instances?|parts?|bits?|occurrences?|momente|szenen|stellen|male?)"
EXHAUSTIVE = (rf"\bevery(?:\s+single)?(?:\s+{GENERIC})?\b|\b(?:all(?: the)?|any|each|jede[nrs]?|alle)\s+{GENERIC}\b|"
              r"\bwhenever\b|\bimmer wenn\b")
# "every shot where ...", "every usable shot of ...": all distinct matches; "every float: A, B, C": one per item.
ALL_MATCHES = (rf"\b(?:every|all(?: the)?|any|each|jede[nrs]?|alle)(?:\s+single)?\s+(?:[^\W\d][\w-]*\s+){{0,2}}?"
               rf"{GENERIC}\b|\bwhenever\b|\bimmer wenn\b")
# "the best moments", "die besten Momente", "highlights" (with no shot list): a generic highlight prior; also "which
# ones should I keep" / "welche ... behalten" (the keepers of a pile of clips).
HIGHLIGHT = (r"\b(?:best|top|greatest|coolest|nicest|beste[nmrs]?|schönste[nmrs]?|coolste[nmrs]?|tollste[nmrs]?)\s+"
             r"(?:[^\W\d][\w-]*\s+)?(?:moments?|bits|parts|highlights|scenes|clips|shots|momente|szenen|stellen)\b|"
             r"\bhighlights?\b|\bkeepers\b|\bworth\s+keeping\b|\b(?:which|welche)\b[^.?!]{0,60}?\b(?:keep|behalten)\b")
# A count with no clip noun, before a relative clause: "die 5 die ich behalten soll", "the 3 that I should keep".
COUNT_RELATIVE = (r"\b(?:the|die|der|das|den|top)\s+(\d+)\s+(?=(?:die|that|which|i|ich|you|du|we|wir|worth|to\s+keep)"
                  r"\b)")
# A reaction brief: the moments it wants are reactions and events of a person on camera (a streamer, a vlogger, a
# player), told by sound and speech more than by the picture: "the funniest moments", "every jump scare, death and
# clutch play", "a cold open with the biggest reaction", "screams", "punchlines". It is a highlight brief whose prior
# leans on the audio reaction and excited speech (vhe.signals.reaction), and its shot list becomes one content text
# (the moments it names are events, not pictures). Read on the brief without its avoid clauses (cue_text).
REACTION_CUE = (r"\b(?:funn(?:y|ier|iest)|hilarious|laugh(?:s|ing|ter|ed)?|jokes?|gags?|punch[\s-]?lines?|banter|"
                r"(?:jump[\s-]?)?scares?|scary|scream(?:s|ing|ed)?|shout(?:s|ing)?|react(?:s|ion|ions|ing|ed)?|rage|"
                r"clutch(?:es)?|fails?|deaths?|dies|died|hype[ds]?|cold[\s-]open|witzig\w*|lustig\w*|lach\w*|"
                r"schrei\w*|reaktion(?:en)?)\b")
# A quoted line ("I made it into Harvard"): matched against the transcript (ParsedQuery.lines) when it has at least
# QUOTE_MIN_WORDS words; a one-word quote ("Granny") is a name or a label, not a line.
QUOTED = r"[\"“„]([^\"“”„]{2,200}?)[\"“”]"
QUOTE_MIN_WORDS = 2
# "in stream order", "chronologically": an ordering rule (clips always come in footage order), not content.
ORDER = (r"\b(?:in|by)\s+(?:the\s+)?(?:stream|chronological|time|story|playback|original|broadcast|footage|"
         r"recording)\s+order\b|\bchronologically\b|\bin\s+chronologischer\s+reihenfolge\b")
# What the footage is, named as the deliverable's subject ("highlights video from this VOD"): no content.
SOURCE_WORDS = {"this", "that", "the", "my", "his", "her", "their", "our", "full", "whole", "entire", "raw", "vod",
                "vods", "stream", "streams", "livestream", "livestreams", "live", "footage", "video", "videos",
                "recording", "recordings", "broadcast", "material", "archive", "session", "playthrough", "clips"}
# "Cut a ... highlights video from this VOD", "make an edit of ...": a deliverable clause names the format and its
# subject (BUDGET_NOUNS ... of/from X), no shot.
DELIVERABLE = (r"^(?:(?:please|also|then)\s+)*(?:cut|make|create|edit|build|put\s+together|assemble)\s+(?:me\s+)?"
               r"(?:a|an|one|the|my)\b")
# "cut all dead air: the menus, him leaving the desk", "skip the boring parts", "no dead air": the list after it
# (after a colon) is what to leave out, preferably (a soft avoid: such items are often no picture, "slow searching").
DEAD_AIR = (r"\b(?:cut|skip|drop|remove|lose|trim|avoid|no|without|ohne)\s+(?:out\s+)?(?:all\s+(?:of\s+)?(?:the\s+)?|"
            r"the\s+|any\s+)?(?:dead[\s-]air|boring\s+(?:parts?|bits?|stuff|stretches|moments)|dull\s+(?:parts?|bits?|"
            r"stuff|stretches|moments)|filler|downtime|down[\s-]time)\b\s*:?")


DASHES = re.compile(r"[‒–—−]")  # en/em dashes and minus sign -> '-' (also used by the CLI's --budget)


# A duration that describes what to leave out or delete, not a length: "2 minutes of nothing", "2 minuten nix".
NOTHING_AFTER = (r"\s*(?:(?:of|lang|long)\s+)?(?:nix|nichts|nothing|gar\s+nichts|garnichts|where\s+nothing|"
                 r"wo\s+(?:nix|nichts)|in\s+denen\s+nichts|of\s+(?:dead\s+air|empty))\b")


# ---- the plan: wanted shots, things to avoid, spoken topics, live sound ---------------------------------------------
LIVE_AUDIO = (r"(?:\bwith\s+)?\b(?:live|natural|ambient|original|real|nat)[\s-]+(?:natural\s+)?(?:sound|audio)\b"
              r"(?:\s*\(\s*not\s+muted\s*\))?|\(\s*not\s+muted\s*\)|\bnot\s+muted\b|\bunmuted\b|\bsound\s+on\b|"
              r"\bnat[\s-]?sound\b|\bmit\s+originalton\b")
SOUNDBITE = rf"(?:{VALUE})?[\s-]*(?:long\s+)?\b(?:sound[\s-]?bites?|soundbites?)\b"
# "the strongest 20-30 s stretch", "the best shot": one clip
SINGLE = (r"\b(?:the|one|a)\s+(?:single\s+)?(?:best|strongest|top|key|most\s+[\w-]+)\s+(?:[^\W\d][\w-]*\s+){0,2}?"
          r"(?:stretch|clip|shot|moment|sequence|scene|take)\b")
AVOID_STRONG = (r"\b(?:leave out|leaving out|avoid(?:ing)?|exclude|excluding|skip(?:ping)?|cut out|"
                r"(?:don't|do not|never) (?:use|include|show))\b")
AVOID_WEAK = (r"\b(?:no|never|without|nothing|ohne|kein|keine|keinen|nichts|niemals|"
              r"nicht(?!\s+(?:zu|so|mehr|länger|kürzer)\b))\b")
# Hardness of an avoid clause: "no / never / absolutely no / leave out / exclude X" forbids X (a hard gate); "avoid",
# "preferably no", "try not to", "if possible" only prefer to (the soft penalty). AVOID_* above decide the clause's
# extent, these its hardness.
SOFT_AVOID = (r"\b(?:avoid(?:ing)?|preferably|ideally|if possible|where possible|try (?:to|not)|rather not|less|fewer|"
              r"too many|too much|möglichst|lieber|eher|wenn möglich|weniger)\b")
HARD_CUE = (r"\b(?:no|not|never|none|nothing|without|exclude|excluding|leave out|leaving out|cut out|skip|don't|"
            r"do not|kein|keine|keinen|nicht|nichts|ohne|niemals|nie|weglassen|weg lassen)\b")
# An avoid item about speech ("the host's talking", "no voice-over") is not a picture: it drops speech-heavy clips.
SPEECH_ITEM = (r"\b(?:talk|talks|talking|speak|speaks|speaking|speech|voice|voices|voice-?over|narration|commentary|"
               r"chatter|dialogue|sprache|reden|redet|stimme|stimmen|gerede|kommentar)\b")
# A montage whose sound is laid on later: "the host's talking (it goes under music)", "music bed to be added
# later", "to sit under a voice-over", "I will lay music over ...". Clips with speech in them would fight that
# track, so they are gated like "no host talking over it" (ParsedQuery.no_speech), unless the brief wants speech.
SPEECH_FREE = (r"\b(?:goes|go|going|sits?|sitting|runs?|plays?|lay|laid|put|placed)\s+under\s+(?:the\s+|a\s+|some\s+)?"
               r"(?:music|song|track|voice-?over|vo|narration)\b|\bmusic\s+bed\b|"
               r"\b(?:lay|laying|put|putting|add|adding)\s+(?:the\s+|some\s+|my\s+)?music\s+(?:over|on|under)\b|"
               r"\b(?:music|voice-?over|vo)\s+(?:will\s+be|is\s+going\s+to\s+be|to\s+be|gets|goes)\s+"
               r"(?:added|laid|recorded|put|on)\b|"
               r"\bunder\s+(?:a\s+|the\s+)?(?:voice-?over|vo|music)\b|"
               r"\bno\s+(?:host\s+|one\s+)?(?:talking|speaking|speech|voices?)\s+over\b|"
               r"\bmusik\s+(?:drunter|darunter|drüber|darüber)\b|\bunter\s+(?:die\s+|der\s+)?musik\b")
# Spoken topics: "talks (directly to his viewers) about ...", "explains ...", "sums up ...", "says that ...".
TOPIC = (r"\b(?:(?:talks?|talking|speaks?|speaking|chats?|chatting)\s+(?:directly\s+|briefly\s+)?"
         r"(?:(?:to|with)\s+(?:the\s+|his\s+|her\s+|their\s+)?[\w-]+\s+)?(?:about|on the subject of)|"
         r"(?:says?|saying|tells?|telling)\s+(?:[\w-]+\s+)?(?:that|about|why|how|what|where)|"
         r"explains?|explaining|mentions?|mentioning|describes?|describing|discuss(?:es)?|discussing|"
         r"sums?\s+up|summing\s+up|summari[sz](?:es|ing|e)|introduces?|introducing|announces?|announcing|"
         r"spricht\s+über|redet\s+über|erzählt|erklärt)\b")
# Instructions about the output, not content: a sentence that starts like this is dropped whole; a clause like
# "for a news package" or "so they can be used as ..." is dropped on its own.
META_SENTENCE = (r"^(?:also\s+)?(?:for\s+each(?:\s+one|\s+clip|\s+of\s+them)?,?\s+)?"
                 r"(?:give|return|provide|include|output|mark|export|note|add)\b.*"
                 r"\b(?:clips?|times?|time\s*codes?|timecodes?|timestamps?|ranges?|(?:in|out)[\s-]?points?|captions?)\b")
META_CLAUSE = (r"^(?:so\s+(?:that\s+)?(?:they|it|we|i|you)\s+can\b|"
               r"(?:with\s+)?(?:the\s+)?(?:exact\s+)?in[\s-]+and[\s-]+out[\s-]+points?\b|(?:and\s+)?rank(?:ed)?\s+(?:them|it)\b|"
               r"to\s+(?:sit|go|play|run|lay)\s+(?:under|over|with)\b|for\s+(?:a|an|the|our|my)\s+[\w\s-]*?\b(?:package|voice-?over|segment|story|piece|post|deck|"
               r"channel|show|report)\b|for\s+example\b|e\.?g\.?$)")
LEAD = (r"^(?:(?:also|please|then|plus|and|or)\s+)*(?:(?:cut|make|create|build(?:\s+up)?(?:\s+through)?|show|include|"
        r"feature|cover|pick|find|get|give(?:\s+me)?|list|use|add|want|need|i\s+want|i\s+need|we\s+need|look\s+for|"
        r"(?:good\s+)?candidates?(?:\s+(?:include|are|could\s+be))?)\b\s*)?")
SHOT_LEAD = (r"^(?:(?:shots?|clips?|scenes?|footage|frames?|moments?|anything|parts?)\s+"
             r"(?:where|when|with|of|that|in which|showing)?\s*)")
SUPERLATIVE = ["the most", "most", "best", "strongest", "nicest", "greatest", "recommended", "single"]
# An item made only of these words is camera style or a generic clip noun ("steady or locked-off framing", "clips").
STYLE = {"steady", "locked-off", "locked", "off", "static", "tripod", "handheld", "framing", "stable", "smooth", "or",
         "and", "camera", "work", "shots", "shot", "clips", "clip", "scenes", "scene", "moments", "moment",
         "stretch", "stretches", "parts", "highlights"}
GENERIC_HEAD = {"stretch", "example", "examples", "thing", "things", "item", "items", "b-roll", "broll", "clip", "shot",
                "scene", "moment", "part", "bit"}
ITEM_EDGE = EDGE_STOP | {"where", "when", "that", "which", "who", "also", "then", "each", "every", "all", "one", "per"}
FRAMING_HEADS = {"close-up", "close-ups", "closeup", "closeups", "wide", "wides", "establishing", "wide-angle"}
FRAMING_WORDS = FRAMING_HEADS | {"close", "ups", "tight", "view", "views", "angle", "lots", "of", "more"}


# Nothing to see: "has to work with no context", "no slow build" (a pacing rule, not a picture).
ABSTRACT = {"something", "anything", "any", "context", "build", "buildup", "build-up", "intro", "setup", "explanation"}


# ---- visual cues: framing, faces, hook, readable text, centring -------------------------------------------------
# Read on the brief minus its opener/closer descriptions and avoid clauses: "no bored faces" asks for no faces, and
# "open on an establishing wide" describes the opener (its relevance picks the wide), not the whole edit. Framing is
# the majority of close vs wide cues ("wide view ... rather than close-ups": wide; a tie: none).
HOOK_CUE = (r"\bhooks?\b|\bhooky\b|\bscroll[\s-]?stopp\w*|\bthumb[\s-]?stopp\w*|\bfirst-(?:shots?|frames?)\b|"
            r"\b(?:in|within)\s+(?:the\s+)?(?:very\s+)?first\s+(?:second|1\s*s\b|(?:two|2|few)\s+seconds)|"
            r"\bgrabs?\s+(?:the\s+)?(?:attention|viewers?|people)\b|\battention[\s-]grabb\w*")
CENTRED_CUE = (r"\bcent(?:er|r)ed\b|\bcent(?:er|re)\s+of\s+(?:the\s+)?frame\b|\bin\s+the\s+(?:middle|cent(?:er|re))\b|"
               r"\bvertical\b|\bcroppable\b|\bcrop(?:s|ped|ping)?\s+(?:it\s+|them\s+)?(?:to|for)\s+(?:a\s+)?"
               r"(?:square|vertical|portrait)\b|\bmittig\b|\bhochkant\b")
FACES_CUE = r"\bfaces?\b|\bgesicht(?:er)?\b"
CLOSE_CUE = (r"\bclose[\s-]?ups?\b|\bcloseups?\b|\bextreme\s+close\b|\btight(?:ly)?\s+(?:shots?|framed|framing)\b|"
             r"\bfill(?:s|ed|ing)?\s+(?:up\s+)?(?:the\s+)?(?:whole\s+|entire\s+|full\s+)?(?:frame|screen)\b|"
             r"\b(?:frame|screen)[\s-]filling\b|"
             r"\b(?:right|straight|directly)\s+(?:at|into|toward|towards|in\s+front\s+of)\s+"
             r"(?:the\s+)?(?:camera|lens)\b|"
             r"\b(?:com|runn|walk|charg|mov|head|march|ris|fly)\w*\s+(?:right\s+|straight\s+)?(?:at|toward|towards|"
             r"into)\s+(?:the\s+)?(?:camera|lens)\b|\bpunchy\b|\bpov\b|\bnahaufnahmen?\b")
WIDE_CUE = (r"\bestablishing\b|\bwides?\b|\blong\s+(?:views?|shots?)\b|\bsense\s+of\s+place\b|\bas\s+a\s+place\b|"
            r"\b(?:street|place|city|avenue|town|location|setting|landscape|venue)\b(?:\s+[\w'-]+){0,3}?\s+(?:is|as)\s+"
            r"(?:the\s+)?main\s+character\b|\btotale\b|\bweitwinkel\w*")
# "rather than close-ups", "not too wide": the cue word names what the brief does not want.
CUE_NEGATION = (r"\b(?:rather\s+than|instead\s+of|not|fewer|less|no\s+more)\s+(?:(?:any|the|a|too|so|many)\s+)?"
                r"(?:[\w-]+\s+){0,2}?(?:close[\s-]?ups?|closeups?|wides?|faces?|tight\s+shots?)\b|"
                # "one establishing wide that shows the avenue": a single shot of the list, not the edit's framing
                r"\b(?:one|a|an|single)\s+(?:[\w-]+\s+){0,2}?(?:establishing(?:\s+(?:wide|shot))?|wides?|"
                r"close[\s-]?ups?)\b")
LEGIBLE_CUE = (r"\blegib\w*|\breadab\w*|\bread(?:s|ing)?\s+(?:the|its|their|clearly)\b|\bon-screen\s+text\b|"
               r"\bexact\s+(?:text|wording)\b|\bspellings?\b|\bfact-check\w*|\blesbar\w*|\bschrift\w*")
# Words a clause can hold besides its cues and still be only an instruction ("keep everything centered cause im
# cropping to vertical", "first clip has to be the hook"): such a clause gives no shot item.
CUE_FILLER = {"keep", "keeping", "make", "sure", "everything", "anything", "cause", "because", "since", "im", "i'm",
              "it", "its", "them", "they", "has", "have", "had", "be", "is", "are", "must", "should", "need", "needs",
              "got", "gotta", "first", "clip", "clips", "shot", "shots", "frame", "frames", "second", "seconds",
              "work", "works", "slow", "build", "edit", "video", "always", "really", "very", "just", "so", "up",
              "subject", "subjects", "square", "crop", "cropping", "cropped"}
# What an LLM answer may add on top of the rules, per field: its value counts only when the (scoped) brief contains
# one of these stems.
CUE_GROUND = {"close": r"close|fill|face|camera|lens|tight|cent(?:er|re)|hook|vertical|pov|punch|nahauf|gesicht",
              "wide": r"wide|establish|place|long view|landscape|panoram|scenery|totale|weitwinkel",
              "want_faces": r"\bfaces?\b|gesicht", "hook": r"hook|first|scroll|grab|attention|thumb",
              "legible_text": r"legib|readab|\bread|text|spell|fact|lesbar|schrift", "centered":
              r"cent(?:er|re)|middle|vertical|crop|square|mittig|hochkant"}
