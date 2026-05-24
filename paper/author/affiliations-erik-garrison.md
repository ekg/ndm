# Affiliation Lookup — Erik Garrison

Task: `affiliation-lookup-erik` · Date: 2026-05-24 · Agent: agent-150

Read-only verification of Erik Garrison's two stated affiliations for the
NDM paper author block: Poietic PBC and the University of Tennessee Health
Science Center (UTHSC). The propagation task (paper edit) is separate.

---

## 1. UTHSC — verified

| Field | Value | Source |
|---|---|---|
| Title | Associate Professor | ORCID employments record (current, no end date); GitHub profile bio; UTHSC News articles |
| Start of current title | 2020-11-01 | ORCID employments record for 0000-0003-3821-631X |
| Department | Department of Genetics, Genomics and Informatics | UTHSC department faculty/staff page; published-paper affiliation blocks |
| Mailing address (dept) | 71 S. Manassas Street, 4th Floor, Memphis, TN 38163, USA | UTHSC GGI department contact information |
| City / State / ZIP | Memphis, TN 38163, USA | Published papers and department page |
| Institutional email | egarris5@uthsc.edu | UTHSC faculty profile |
| Personal email (paper) | erik.garrison@gmail.com | Per task description |
| ORCID | 0000-0003-3821-631X | pub.orcid.org/v3.0/0000-0003-3821-631X/employments |

### Canonical institutional affiliation string

The form used in Erik Garrison's recently co-authored peer-reviewed papers:

> Department of Genetics, Genomics and Informatics, University of Tennessee Health Science Center, Memphis, TN 38163, USA

A minor Oxford-comma variant — `Department of Genetics, Genomics, and Informatics, ...` — appears in some PMC entries. The no-Oxford-comma form
matches the official department name on the UTHSC site and is recommended.

### Note on "College of Medicine"

GGI sits under UTHSC's College of Medicine in the university's organizational
chart, but Erik's recent paper affiliation blocks do not include "College of
Medicine" as a nesting level. The recommended string above follows that
established convention. `[author may confirm whether College of Medicine should be inserted]`

### Sources (UTHSC)
- ORCID employments API (canonical): https://pub.orcid.org/v3.0/0000-0003-3821-631X/employments
- UTHSC faculty page: https://www.uthsc.edu/faculty/profile/?netid=egarris5
- UTHSC GGI department: https://www.uthsc.edu/genetics/ and https://www.uthsc.edu/genetics/faculty-staff.php
- GitHub profile (bio: "Associate professor at the University of Tennessee Health Science Center / bioinformatician / sequence analysis and pangenomes"): https://github.com/ekg
- Panacus PMC paper (affiliation block): https://pmc.ncbi.nlm.nih.gov/articles/PMC11195249/
- Rat pangenome PMC paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10802574/

---

## 2. Poietic PBC — author-provided

Poietic PBC is a new affiliation; per author clarification, it may not yet
have substantial public web presence. Use the author-provided name "Poietic PBC"
as-is in the author block. Brief checks (web search, his GitHub bio, ORCID
employments, his personal blog) returned no public listing — this is expected
for a newly-formed entity and is not a verification failure.

| Field | Value | Note |
|---|---|---|
| Entity name | Poietic PBC | `[author-provided]` — use verbatim |
| State of incorporation | — | `[author-provided if needed]` — omit from author block unless author supplies |
| Mailing address | — | `[author-provided if needed]` — omit unless author supplies |
| Erik's role | — | `[author-provided if needed]` — not required in the author block itself |
| Website / URL | — | `[author-provided if any]` |

The author block should list "Poietic PBC" without a location line until/unless
the author supplies one. Do not invent a city or state.

---

## 3. Typst author-block snippets

Both styles use:
- Name: Erik Garrison
- Email: erik.garrison@gmail.com (per task)
- ORCID: 0000-0003-3821-631X (verified)
- Poietic PBC: name as given by author, no location appended
- UTHSC: full verified institutional string

### Style A — single author, inline affiliations

```typst
#align(center)[
  #text(size: 14pt, weight: "bold")[Erik Garrison]
  #v(0.3em)
  #text(size: 10pt)[
    Poietic PBC \
    Department of Genetics, Genomics and Informatics, \
    University of Tennessee Health Science Center, Memphis, TN 38163, USA \
    #link("mailto:erik.garrison@gmail.com")[erik.garrison\@gmail.com] · ORCID: #link("https://orcid.org/0000-0003-3821-631X")[0000-0003-3821-631X]
  ]
]
```

### Style B — numbered affiliations, NeurIPS-style

```typst
#align(center)[
  #text(size: 14pt, weight: "bold")[Erik Garrison#super[1,2]]
  #v(0.3em)
  #text(size: 10pt)[
    #super[1] Poietic PBC \
    #super[2] Department of Genetics, Genomics and Informatics, University of Tennessee Health Science Center, Memphis, TN 38163, USA \
    #link("mailto:erik.garrison@gmail.com")[erik.garrison\@gmail.com] · ORCID: #link("https://orcid.org/0000-0003-3821-631X")[0000-0003-3821-631X]
  ]
]
```

### Drop-in replacement for current title block in `paper/main.typ`

The current title block at `paper/main.typ:49-59` shows:

```typst
#text(size: 11.5pt)[Erik Garrison]
#v(0.2em)
#text(size: 9.5pt, style: "italic")[Independent]
```

A minimally invasive Style-A replacement keeping the current sizing:

```typst
#text(size: 11.5pt)[Erik Garrison]
#v(0.2em)
#text(size: 9.5pt)[
  Poietic PBC \
  Department of Genetics, Genomics and Informatics, University of Tennessee Health Science Center, Memphis, TN 38163, USA \
  #link("mailto:erik.garrison@gmail.com")[erik.garrison\@gmail.com] · ORCID #link("https://orcid.org/0000-0003-3821-631X")[0000-0003-3821-631X]
]
```

---

## 4. Validation checklist

- [x] UTHSC affiliation verified: title, department, full institutional string, city/state
- [x] Poietic PBC handled per author clarification: name used as-is, marked `[author-provided]`, no location guessed
- [x] Author-block snippet produced in BOTH Style A (inline) and Style B (numbered) Typst formats
- [x] Email included (erik.garrison@gmail.com)
- [x] ORCID included (0000-0003-3821-631X, verified)
- [x] Sources cited for UTHSC (faculty page, ORCID API, PMC papers)
- [x] Unverifiable/author-provided fields are marked rather than guessed

## 5. Recommended follow-up before the propagation task

Before the next agent edits `paper/main.typ`, Erik may want to confirm:

1. Final spelling preference for the new entity: "Poietic PBC" (default), or
   an alternate formal form such as "Poietic, PBC".
2. Whether a Poietic PBC location line should appear on the author block.
3. Style preference: Style A (inline) vs Style B (numbered superscripts).
   Style B is the cleanest form when both affiliations should be attached to
   the single author with formal markers.
