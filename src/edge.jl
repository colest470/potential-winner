 import Pkg;

 Pkg.add("Images")
 Pkg.add("Colors")

 using Images
 using Colors

function detectEdges(file_path)
    image = load(file_path)

    result = fill(RGB(0, 0, 0), size(image))

    for y in axes(image, 1)
        for x in axes(image, 2)
            p = image[y, x]
            if red(p) > 0.4 && green(p) > 0.4 && blue(p) > 0.3
                result[y, x] = RGB(0, 0, 0)
            else
                result[y, x] = RGB(1, 1, 1)
            end
        end
    end

    return channelview(result)
end