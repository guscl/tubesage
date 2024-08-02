import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TextSplitter:
    def split(self, text: str) -> List[str]:
        pass


class LangChainSmartTextSplitter(TextSplitter):
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        """
        Initialize LangCahinSmartTextSplitter.
        Tries to keep all paragraphs (and then sentences, and then words) together as long as possible

        Args:
            chunk_size (int): The size of the chunks to split the text into.
            chunk_overlap (int): The overlap between the chunks.
        """
        try:
            self.text_spliter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as e:
            logger.error(f"Failed to initialize RecursiveCharacterTextSplitter: {e}")
            raise

    def split(self, text: str) -> List[str]:
        """
        Splits the given text into a smaller .

        Args:
            text (str): The text to be split.

        Returns:
            List[float]: A list of floats representing the splits of the text.

        Raises:
            Exception: If an error occurs while splitting the text.
        """
        try:
            splits = self.text_spliter.split_text(text)
            logger.info("Successfully split text")
            return splits
        except Exception as e:
            logger.error(f"An error occurred while splitting text: {e}")
            raise


if __name__ == "__main__":
    client = LangChainSmartTextSplitter()

    text = """hello my name is andre and i've been training deep neural networks for a bit more than a decade and in this lecture i'd like 
    to show you what neural network training looks like under the hood so in particular we are going to start with a blank jupiter notebook and by 
    the end of this lecture we will define and train in neural net and you'll get to see everything that goes on under the hood and exactly sort of 
    how that works on an intuitive level now specifically what i would like to do is i would like to take you through building of micrograd now micrograd is 
    this library that i released on github about two years ago but at the time i only uploaded the source code and you'd have to go in by yourself and really 
    figure out how it works so in this lecture i will take you through it step by step and kind of comment on all the pieces of it so what is micrograd and why is 
    it interesting good um micrograd is basically an autograd engine autograd is short for automatic gradient and really what it does is it implements backpropagation now 
    backpropagation is this algorithm that allows you to efficiently evaluate the gradient of some kind of a loss function with respect to the weights of a neural network 
    and what that allows us to do then is we can iteratively tune the weights of that neural network to minimize the loss function and therefore improve the accuracy 
    of the network so back propagation would be at the mathematical core of any modern deep neural network library like say pytorch or jaxx so the functionality of 
    microgrant is i think best illustrated by an example so if we just scroll down here you'll see that micrograph basically allows you to build out mathematical 
    expressions and um here what we are doing is we have an expression that we're building out where you have two inputs a and b and you'll see that a and b are 
    negative four and two but we are wrapping those values into this value object that we are going to build out as part of micrograd so this value object will wrap the 
    numbers themselves and then we are going to build out a mathematical expression here where a and b are transformed into c d and eventually e f and g and i'm showing 
    some of the functions some of the functionality of micrograph and the operations that it supports so you can add two value objects you can multiply them you can raise them to a 
    constant power you can offset by one negate squash at zero square divide by constant divide by it etc and so we're building out an expression graph with with these two inputs 
    a and b and we're creating an output value of g and micrograd will in the background build out this entire mathematical expression so it will for example know that c is 
    also a value c was a result of an addition operation and the child nodes of c are a and b because the and will maintain pointers to a and b value objects so we'll 
    basically know exactly how all of this is laid out and then not only can we do what we call the forward pass where we actually look at the value of g of course that's 
    pretty straightforward we will access that using the dot data attribute and so the output of the forward pass the value of g is 24.7 it turns out but the big deal is 
    that we can also take this g value object and we can call that backward and this will basically uh initialize back propagation at the node g and what backpropagation is 
    going to do is it's going to start at g and it's going to go backwards through that expression graph and it's going to recursively apply the chain rule from calculus and 
    what that allows us to do then is we're going to evaluate basically the derivative of g with respect to all the internal nodes like e d and c but also with respect to the 
    inputs a and b and then we can actually query this derivative of g with respect to a for example that's a dot grad in this case it happens to be 138 and the derivative of g 
    with respect to b which also happens to be here 645 and this derivative we'll see soon is very important information because it's telling us how a and b are affecting g through this 
    mathematical expression so in particular a dot grad is 138 so if we slightly nudge a and make it slightly larger 138 is telling us that g will grow and the slope of that growth is going to 
    be 138 and the slope of growth of b is going to be 645. so that's going to tell us about how g will respond if a and b get tweaked a tiny amount in a positive direction okay now you might be"""

    splits = client.split(text)
    logger.info(splits)
