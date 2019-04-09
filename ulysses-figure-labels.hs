import Text.Pandoc.JSON

main:: IO()
main = toJSONFilter extractFigLabel
    where
        -- FIXME: I don't know what the options are
    extractFigLabel (Image (id, classes, options)
                     caption (url, title)) | id == "" =
        Image (title, classes, options) caption (url,[])
    extractFigLabel x = x


