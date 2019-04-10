import Text.Pandoc.JSON

main:: IO()
main = toJSONFilter extractFigLabel
    where
        extractFigLabel (Image (id, classes, options)
                     caption (url, title)) =
            Image (title, classes, options) caption (url,"fig:")
        extractFigLabel x = x


