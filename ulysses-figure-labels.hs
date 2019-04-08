import Text.Pandoc.JSON

main:: IO()
main = toJSONFilter extractFigLabel
    where
    extractFigLabel (Image (id, classes, keyValPairs)
                     caption (url, title)) =
        Image (title, classes, keyValPairs) caption (url,[])
    extractFigLabel x = x


