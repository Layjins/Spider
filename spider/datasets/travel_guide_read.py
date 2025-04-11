import json
import random

# Load JSON data from a file
with open('travel_guide.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# read_mode = 'random_one'
read_mode = 'all'

if read_mode == 'random_one':
  #####################
  # Randomly select one travel guide
  random_guide = random.choice(data['travel_guide'])

  # Get the location
  location = random_guide['location']
  print(f"Location: {location}")

  answer_multimodal = random_guide['answer_multimodal']

  # Print the content of the randomly selected travel guide using answer_multimodal
  for section, content in answer_multimodal.items():
          print(f"\n{section.replace('_', ' ').title()}:")
          if section == 'introduction':
              print(f"{content}")
          elif section == 'closing':
              print(f"{content}")
          else:
            for item in content:
                print(f" - {item}")

  print("\n" + "="*50 + "\n")
  #####################
else:
  #####################
  # Iterate over each location and print the travel guide
  for guide in data['travel_guide']:
      location = guide['location']
      print(f"Location: {location}")
      
      answer_multimodal = guide['answer_multimodal']
      
      # Print the content of the travel guide using answer_multimodal
      for section, content in answer_multimodal.items():
          print(f"\n{section.replace('_', ' ').title()}:")
          if section == 'introduction':
              print(f"{content}")
          elif section == 'closing':
              print(f"{content}")
          else:
            for item in content:
                print(f" - {item}")
      
      print("\n" + "="*50 + "\n")
  print('Number of cities:', len(data['travel_guide'])) # 135 cities
  #####################
