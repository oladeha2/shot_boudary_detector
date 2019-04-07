# crop video of any size to 64 x 64 required for input to the model
def six_four_crop_video(newVideo):
    """
    Args:
        newVideo --> type: moviepy VideoFileClip
    Funtionality:
        pass in video of any size. Result is a video transformed and resized to the ideal
        64 x 64 resolution required for input to the convolutional neural network
    """
    dimensions = newVideo.size
    newVideo.reader.close()
    smaller_dimension = min(dimensions)
    target = 64
    factor = smaller_dimension/target
    factored_dimensions = []
    for i in range(2):
        factored_dimensions.append(round(dimensions[i]/factor))
    first_stage_crop = newVideo.resize((factored_dimensions[0], factored_dimensions[1]))
    larger_dimension = max(factored_dimensions)
    midpoint = round(larger_dimension/2)
    limit = target/2
    lower = midpoint - limit #x1
    upper = midpoint + limit #x2
    six_four_crop = first_stage_crop.crop(x1=lower, y1=0, x2=upper, y2=64)
    print('final size: ', six_four_crop.size)
    return six_four_crop