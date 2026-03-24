import pandas as pd
import pickle
import difflib

print("Loading app...")
movies     = pd.read_pickle('model/movies.pkl')
knn        = pickle.load(open('model/knn_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
print("Ready!\n")

def recommend(title):
    matches = movies[movies['title'].str.lower() == title.lower()]

    if matches.empty:
        print(f"\n❌ Movie '{title}' not found.")
        
        all_titles = movies['title'].tolist()
        suggestions = difflib.get_close_matches(title.title(), all_titles, n=3, cutoff=0.5)
        
        if suggestions:
            print("Did you mean:")
            for s in suggestions:
                print(f"  → {s}")
        return

    idx = matches.index[0]
    movie_title = movies.iloc[idx]['title']
    movie_tags = movies.iloc[idx]['tags']

    target_vector = vectorizer.transform([movie_tags])
    distances, indices = knn.kneighbors(target_vector)

    print(f"\n🎬 Because you liked '{movie_title}', you might enjoy:\n")
    
    for rank, i in enumerate(indices[0][1:6], start=1):
        score = round((1 - distances[0][rank]) * 100, 1)
        rec_title = movies.iloc[i]['title']
        print(f"  {rank}. {rec_title} (similarity: {score}%)")

def main():
    print("=" * 45)
    print("       🎥 Movie Recommendation System")
    print("=" * 45)

    while True:
        print("\nOptions: type a movie name | 'quit' to exit")
        user_input = input("\nEnter movie name: ").strip()

        if user_input.lower() == 'quit':
            print("\nGoodbye! 🎬")
            break
            
        if user_input == "":
            continue

        recommend(user_input)

if __name__ == '__main__':
    main()
