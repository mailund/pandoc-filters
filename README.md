# pandoc-filters

Filters you can use with Pandoc.

The `ulysses-figure-labels` filter puts a figure’s title (not its caption) as a label. If you export a figure from Ulysses it will look like this:

```
  ![A figure text](many-to-many-compilation.pdf "my_test")
```

where “a figure text” is the caption you have in the Ulysses file and “my_test” is the title. The filter translates it into

```
  ![A figure text](many-to-many-compilation.pdf){#fig:my_test}
```
  
This format is exactly what `pandoc-crossref` can use.

