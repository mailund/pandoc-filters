---
header-includes: |
    \newcounter{exercounter}[section]
    \newcommand{\theexercise}{\thesection.\arabic{exercounter}}
    \makeatletter
    \newenvironment{exercises}{%
    \par\refstepcounter{exercounter}\protected@edef\@currentlabel{\theexercise}%
    \noindent\textbf{Exercise \theexercise}}{}
    \makeatother

---



[[See Exercise 1 @ex:ex1]]{.out .html} [[See @ex:ex2]]{.out .latex}


::: {#ex:ex1 .Exercise}
First exercise
:::

::: {#ex:ex2 .Exercise}
Second exercise
:::
