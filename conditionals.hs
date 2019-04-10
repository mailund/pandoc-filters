import Text.Pandoc.JSON

main:: IO()
main = toJSONFilter reform
    where
        reform (CodeBlock  header  body) =  --[Para [body]]
            Image (Str "foo", [Str"title"], [(Str "classes", Str "or not")])
        --reform x = x
