Build one complete file, index.html unless the user names another. Indented, never minified.
Head: <!doctype html>; html lang = the user's language (pt-BR or en); meta charset; meta viewport width=device-width, initial-scale=1; one <style>. No external libraries, fonts or CDNs.
Body: header; main with one <section id="..."> per requested section, in order; footer. Exactly the number of items and sentences the user asked for.
Content: specific, believable text in the user's language about THIS business and its city: description, product names, prices in that city's currency. Never Lorem ipsum, "Item 1", "Texto aqui".
Address, phone or email the user did not give: a visible placeholder in that country's format, with the word Exemplo or Example and zeros for the digits.
Footer: © and the current year from TEMPORAL CONTEXT.
Colors: the colors the user asked for. Light page, dark text. Every colored block sets background AND a strongly contrasting text color (dark on light, light on dark); never text near its background color.
Layout: body margin 0; sections padding 2rem 1rem; text max-width 70ch, centred; img max-width 100%; line-height 1.6; serif titles if asked; plus one @media (min-width:768px) block.
Call to action: <a class="btn" href="#section-id">, display:inline-block, padding .75rem 1.25rem, rounded, no underline, min-height 44px. Never <a> inside <button>.
Code-only request: one fenced block, nothing else.
