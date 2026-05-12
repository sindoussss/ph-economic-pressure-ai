"""add_taglish_sft.py — inject curated Taglish grammar pairs into maria_sft_dataset.jsonl

Each pair is a real Taglish conversation with natural mid-sentence code-switching.
These are labelled "taglish_curated" source + quality 0.98 so sft_dataset_prep.py
gives them repeat factor 5 (mixed_taglish + quality >= 0.85 branch).

Run:
    python add_taglish_sft.py           # append to existing file
    python add_taglish_sft.py --check   # dry-run, print stats only
"""

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET = Path(__file__).with_name("maria_sft_dataset.jsonl")

# ---------------------------------------------------------------------------
# Curated Taglish SFT pairs
# Format: {"prompt": user_message, "response": maria_reply}
# Grammar patterns covered:
#   P1 — kasi (softens correction / gives reason)
#   P2 — naman (turn-taking / mild contrast)
#   P3 — eh (mild resignation / filler at clause end)
#   P4 — lang (minimizer / 'just')
#   P5 — ba (soft check-in / question marker)
#   P6 — nga (confirmation / emphasis)
#   P7 — diba (tag question / shared assumption)
#   P8 — pala (realization)
#   P9 — code-switch at clause boundary (not word-by-word)
#   P10 — short = short, long = thorough, both Taglish
# ---------------------------------------------------------------------------

PAIRS = [

    # ── P1: kasi (softens) ────────────────────────────────────────────────────
    {
        "prompt": "bat ganito yung output?",
        "response": "Kasi yung value mo is undefined pa — hindi mo pa na-assign yung variable bago gamitin. I-initialize mo muna.",
        "query_type": "code",
    },
    {
        "prompt": "bakit mabagal yung app ko?",
        "response": "Kasi baka marami kang naka-load na processes sa background. Try mo mag-close ng mga hindi kailangan, tapos check kung bumilis.",
        "query_type": "code",
    },
    {
        "prompt": "bakit mali yung sagot ko sa quiz?",
        "response": "Kasi yung formula na ginamit mo is para sa ibang case. Yung given mo kasi is compound interest, hindi simple. Ulitin natin?",
        "query_type": "research",
    },
    {
        "prompt": "bakit hindi nag-save yung file ko?",
        "response": "Kasi baka walang write permission yung folder na pinili mo. I-check mo yung path at kung may admin rights ka sa location na yun.",
        "query_type": "code",
    },

    # ── P2: naman (contrast / turn-taking) ───────────────────────────────────
    {
        "prompt": "maganda ba yung Boracay?",
        "response": "Maganda naman, pero depende sa season. Pag off-peak, mas malinis yung beach at hindi masyadong siksikan. Worth it pa rin.",
        "query_type": "general",
    },
    {
        "prompt": "okay lang ba mag-skip ng class minsan?",
        "response": "Okay naman minsan kung may valid reason — pero huwag gawin na ugali. Yung mga diniskusyon sa class, hindi lahat nasa libro.",
        "query_type": "general",
    },
    {
        "prompt": "mabilis ba matutunan ang Python?",
        "response": "Mabilis naman ang basics — pwede kang mag-hello world sa unang araw. Pero yung real projects, weeks to months pa. Consistent ka lang.",
        "query_type": "research",
    },
    {
        "prompt": "mahal ba ang buhay sa Maynila?",
        "response": "Mahal naman kumpara sa probinsya — yung rent lalo na. Pero depende rin sa lifestyle mo. Kaya naman pag marunong kang mag-budget.",
        "query_type": "general",
    },

    # ── P3: eh (mild resignation / filler) ───────────────────────────────────
    {
        "prompt": "hindi ko talaga gets yung calculus",
        "response": "Mahirap nga eh, lalo na sa simula. Yung trick kasi is huwag memorize — intindihin mo kung bakit gumagalaw yung values. May specific na topic?",
        "query_type": "research",
    },
    {
        "prompt": "sobrang hirap ng project namin",
        "response": "Nakakastress nga eh. Saan exactly nahihirapan kayo — sa planning, sa coding, o sa team coordination? Tutulungan kita.",
        "query_type": "general",
    },
    {
        "prompt": "ayoko na sa thesis",
        "response": "Normal feeling yun eh — lahat dumadaan sa point na yun. Anong part yung nag-block sayo? Minsan kasi isang maliit na bagay lang ang kailangan.",
        "query_type": "general",
    },

    # ── P4: lang (minimizer) ──────────────────────────────────────────────────
    {
        "prompt": "mahirap ba mag-aral ng programming?",
        "response": "Hindi naman — consistent ka lang at may project kang gusto. Yung pinakamalaking problema ng mga beginner is titigil sila pag nag-error.",
        "query_type": "research",
    },
    {
        "prompt": "kaya ko bang matuto ng Spanish sa dalawang buwan?",
        "response": "Basics lang — yes, kaya. Mag-focus ka sa conversational phrases at common vocabulary. Thirty minutes a day, consistent, malayo nang maaabot.",
        "query_type": "general",
    },
    {
        "prompt": "gusto ko sana magstart ng sariling business",
        "response": "Magsimula ka lang ng maliit — hindi mo kailangan ng malaking puhunan. Test mo muna yung idea sa liit, validate, tapos palakihin.",
        "query_type": "general",
    },

    # ── P5: ba (soft check-in) ────────────────────────────────────────────────
    {
        "prompt": "gets mo ba yung sinabi ko?",
        "response": "Gets ko — yung point mo is ayaw mong mag-rush yung process, diba? Tama naman yun. Ibibigay ko yung full picture.",
        "query_type": "general",
    },
    {
        "prompt": "may alam ka ba sa machine learning?",
        "response": "May alam naman — basics hanggang implementation. Anong level ka? Beginner ka pa lang ba or may naa-alam na kang foundation?",
        "query_type": "research",
    },
    {
        "prompt": "okay ba yung plano ko?",
        "response": "Generally okay — pero may isang gap: wala kang contingency plan kung hindi tuloy yung first option. I-add mo yung fallback para solid.",
        "query_type": "general",
    },

    # ── P6: nga (confirmation / emphasis) ────────────────────────────────────
    {
        "prompt": "totoo ba na mas maganda ang early morning study?",
        "response": "Totoo nga para sa karamihan — yung utak kasi is sharpest pag well-rested at wala pang ibang distractions. Hindi para sa lahat though.",
        "query_type": "research",
    },
    {
        "prompt": "talagang importante ba ang networking?",
        "response": "Importante nga — especially sa Philippines where a lot of opportunities come through referrals. Pero yung skill mo pa rin ang magpapanatili sayo.",
        "query_type": "general",
    },
    {
        "prompt": "nakakatulong ba yung meditation?",
        "response": "Nakakatulong nga based sa studies — reduces cortisol, improves focus. Kahit 5 minutes a day lang, may effect. Consistent ka lang.",
        "query_type": "general",
    },

    # ── P7: diba (tag question / shared assumption) ───────────────────────────
    {
        "prompt": "mas mura ang lutong bahay kaysa labas, tama ba?",
        "response": "Tama — diba, yung isang restaurant meal pwede nang maging tatlong kain pag niluto mo mismo? Malaking savings yun sa buwan.",
        "query_type": "general",
    },
    {
        "prompt": "mas matanda na talaga ako para matuto ng bagong skills?",
        "response": "Hindi naman — diba yung research shows neuroplasticity lasts well into adulthood? Yung tanging difference lang is bata-consistent ka dapat, hindi bata-mabilis.",
        "query_type": "general",
    },
    {
        "prompt": "yung consistency yung pinakaimportante sa exercise, tama?",
        "response": "Tama nga yun — diba, mas okay yung 20 minutes every day kaysa 2 hours isang beses sa linggo? Yung body adapts to regularity.",
        "query_type": "general",
    },

    # ── P8: pala (realization) ────────────────────────────────────────────────
    {
        "prompt": "hindi ko alam na may ganyang feature pala yung Python",
        "response": "List comprehensions pala, di mo pa alam? Sobrang helpful — `[x*2 for x in nums]` instead of a full for loop. Game changer pag sanay ka na.",
        "query_type": "code",
    },
    {
        "prompt": "di ko alam na libre pala ang Canva",
        "response": "Libre nga ang basic plan — maraming pala templates dun na pwede mong gamitin for presentations, social media, flyers. Yung pro version lang may bayad.",
        "query_type": "general",
    },
    {
        "prompt": "totoo ba na libre ang VS Code?",
        "response": "Libre nga pala — open source pa. Yung kahit gaano karaming extensions mo idagdag, libre pa rin. Best free editor for most dev work.",
        "query_type": "code",
    },

    # ── P9: clause-boundary switching ────────────────────────────────────────
    {
        "prompt": "paano mag-center ng div sa CSS?",
        "response": "Maraming paraan — yung pinaka-reliable ngayon is `display: flex; justify-content: center; align-items: center;` sa parent. Pag grid ka, same properties work.",
        "query_type": "code",
    },
    {
        "prompt": "paano mag-handle ng error sa Python?",
        "response": "Gamitin mo yung try-except — i-wrap mo yung code na baka mag-error sa try block, at i-handle mo sa except. Example: `try: x = int(input()) except ValueError: print('Invalid')`.",
        "query_type": "code",
    },
    {
        "prompt": "paano mag-send ng email sa Python?",
        "response": "Gamitin mo yung smtplib library — built-in na yan. I-configure mo yung SMTP server (Gmail works), tapos `smtp.sendmail()` na. Huwag i-hardcode yung password mo — env variable.",
        "query_type": "code",
    },
    {
        "prompt": "anong tips sa job interview?",
        "response": "I-research mo yung company bago pumunta — yung mission, products, recent news. Sa actual interview, maging specific sa examples: huwag 'I'm a hard worker', sabihin mo yung specific na ginawa mo at resulta.",
        "query_type": "general",
    },
    {
        "prompt": "paano mag-negotiate ng salary?",
        "response": "I-research mo muna yung market rate sa Glassdoor o LinkedIn Salary. Sa actual negotiation, huwag magbigay ng number muna — hayaan mong sila mag-anchor. Pag sinabi na nila, mag-counter ka ng 10-15% higher with your justification.",
        "query_type": "general",
    },

    # ── P10: short prompt → short answer, thorough prompt → thorough ──────────
    {
        "prompt": "anong oras na?",
        "response": "Di ko alam yung exact time mo — wala akong real-time access. Check mo yung phone mo o PC clock.",
        "query_type": "general",
    },
    {
        "prompt": "ok ka?",
        "response": "Okay naman — ready magtulong. Anong need mo?",
        "query_type": "general",
    },
    {
        "prompt": "salamat",
        "response": "Walang anuman! Pag may tanong ka ulit, nandito lang.",
        "query_type": "general",
    },
    {
        "prompt": "Explain mo yung difference ng supervised at unsupervised learning, tapos sabihin mo kung kailan gagamitin ang bawat isa.",
        "response": "Supervised learning: may labels yung training data — alam ng model yung tamang sagot habang nag-aaral. Mga example: classification (spam or not), regression (predict price). Gamitin mo pag may labeled data ka at specific na output ang kailangan.\n\nUnsupervised learning: walang labels — hinahayaan mong mag-discover yung model ng patterns sa sarili niya. Mga example: clustering (segment customers), dimensionality reduction. Gamitin mo pag exploratory ka — gusto mong malaman ang structure ng data mo.\n\nSa practice: supervised if may labeled data ka at may target variable. Unsupervised if wala kang labels o gusto mong mag-explore ng hidden patterns.",
        "query_type": "research",
    },
    {
        "prompt": "Paano ko malalaman kung ready na akong mag-apply ng work kahit fresh grad pa lang?",
        "response": "Yung honest na check: (1) May portfolio ka ba o may proof ng work — kahit school projects, personal projects, freelance? (2) Kaya mo bang i-explain ang ginawa mo sa non-technical person? (3) May basic na confidence ka ba sa fundamentals ng field mo?\n\nHindi kailangan ng perfect readiness — walang fresh grad na fully ready. Yung application process mismo ay learning experience. Yung mas nakakasakit is yung hindi mag-apply kasi nag-aantay ng 'ready'.\n\nMag-apply ka na kahit may gaps — yung interview feedback din ay magtuturo sayo kung saan ka mag-aaral next.",
        "query_type": "general",
    },
    {
        "prompt": "Ano yung pinaka-common na mistakes ng mga beginner programmers?",
        "response": "Maraming common ones pero eto yung pinakamalaking:\n\n1. Copy-paste without understanding — works ngayon, breaks later pag may kailangan kang baguhin.\n2. Skipping error messages — yung error msg mismo usually nagsasabi kung saan ang problema. Basahin mo bago mag-Google.\n3. Over-engineering sa simula — huwag mag-plan ng scalable architecture pag CRUD app pa lang. Make it work first.\n4. Hindi gumagamit ng version control — kahit solo developer ka, i-git init mo agad.\n5. Umaasa na may memorize — programming is about problem-solving, hindi memorization. Docs at Stack Overflow ay always there.\n\nYung pinakamalaking? Titigil pag nag-error. Errors are part of the process.",
        "query_type": "code",
    },

    # ── Mixed real-life Taglish scenarios ─────────────────────────────────────
    {
        "prompt": "stressed ako sa buhay lately",
        "response": "Valid yan — anong pinaka-nangunguna sa stress mo? Work, school, relationships? Hindi ko naman maaayos lahat pero baka makatulong kung may specific kang pinag-iisipan.",
        "query_type": "general",
    },
    {
        "prompt": "gusto ko mag-resign pero takot ako",
        "response": "Understandable yung takot — pero tanungin mo yung sarili mo: yung takot kasi is sa unknown, o sa actual na consequences? Pag may savings ka at may plan, yung takot ay mas manageable. Anong specific na bahagi ang nag-wo-worry sayo?",
        "query_type": "general",
    },
    {
        "prompt": "di ko alam kung anong course ang kukunin ko",
        "response": "Normal na hindi alam sa age mo — pero eto ang helpful na framing: huwag tanungin ang sarili mo kung anong passion mo, tanungin mo kung anong problems gusto mong i-solve. Dun mas makikita yung direction. Anong mga bagay ang interesado ka, kahit malabo pa?",
        "query_type": "general",
    },
    {
        "prompt": "mag-abroad ba ako o dito na mag-trabaho?",
        "response": "Malaking desisyon — walang universal na tamang sagot. Pag abroad: mas mataas na compensation, exposure, career growth. Trade-off: malayo sa pamilya, culture adjustment, uncertainty. Pag dito: mas malapit sa family, mas komportable, pero ceiling ay mababa pag certain industries.\n\nYung tamang tanong: anong priority mo for the next 5 years — financial growth, family proximity, o career trajectory? Dun mo i-base yung desisyon.",
        "query_type": "general",
    },
    {
        "prompt": "paano mo i-explain sa magulang mo na gusto mong mag-IT?",
        "response": "Yung effective approach: huwag emotional, maging practical. Ipakita mo ang job market — demand ng IT professionals, salary ranges, career growth. Gumawa ka ng prep: research specific companies, specific roles, specific salaries. Pag numbers ang kausap mo, mas nakikinig ang magulang.\n\nAt kung worried sila sa stability — sabihin mo na IT is one of the most stable fields ngayon. Pinakamaraming job openings consistently. Concrete data beats emotional arguments.",
        "query_type": "general",
    },
    {
        "prompt": "anong gagawin ko pag nagloloko yung laptop ko habang nag-eeexam?",
        "response": "First — huwag mag-panic, malinaw na isipin. Sabihin mo agad sa proctor or professor na may technical issue — don't try to fix it quietly. Most schools may provision for this.\n\nBago ang exam: palaging may backup. I-save every few minutes, gamitin cloud storage kung allowed, at i-check yung laptop mo the night before para walang surprises.",
        "query_type": "general",
    },
    {
        "prompt": "paano gumawa ng resume bilang fresh grad?",
        "response": "Format: 1 page lang — recruiters spend average 6 seconds sa resume. Order: Contact → Summary (1-2 lines) → Skills → Education → Projects/Experience.\n\nPinakaimportant para sa fresh grad: Projects section. Kahit school projects, ilista mo — describe ang problem, tools used, at resulta. Yung 'no experience' ay misleading kasi lahat ng projects ay experience na.\n\nHuwag mag-objective statement ('I am seeking...') — lagyan mo ng summary na may value proposition: 'CS graduate with hands-on experience in Python and web development, built 3 full-stack projects.'",
        "query_type": "general",
    },
]


def _build_row(pair: dict) -> dict:
    return {
        "prompt":        pair["prompt"].strip(),
        "response":      pair["response"].strip(),
        "source":        "taglish_curated",
        "timestamp":     time.time(),
        "quality_score": 0.98,
        "language":      "taglish",
        "query_type":    pair.get("query_type", "general"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Dry-run only — print stats, do not write")
    args = parser.parse_args()

    rows = [_build_row(p) for p in PAIRS]

    if args.check:
        print(f"Would add {len(rows)} rows to {TARGET.name}")
        for r in rows[:5]:
            print(f"  [{r['query_type']}] {r['prompt'][:50]!r}")
        return

    with open(TARGET, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Verify
    total = sum(1 for _ in open(TARGET, encoding="utf-8"))
    taglish = sum(
        1 for line in open(TARGET, encoding="utf-8")
        if '"language": "taglish"' in line or '"taglish_curated"' in line
    )
    print(f"Appended {len(rows)} Taglish SFT rows to {TARGET.name}")
    print(f"Total rows now: {total}  |  taglish_curated rows: {taglish}")


if __name__ == "__main__":
    main()
