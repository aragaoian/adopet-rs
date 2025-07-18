<div align="center">
  <img 
    src="https://github.com/user-attachments/assets/705f8332-ed9e-48d9-9126-56a733c19a29" 
    alt="Adopet Logo" 
    style="width: 50%; height: auto; margin-bottom: 10px;" 
  />
  <p style="max-width: 600px;">
    <b>ADOPET Rs</b>: A Feature-Weighted Content-Based filtering developed to decrease adopted pets abandonment and aid to democratize the adoption process
  </p>
  <a href="https://adopet.top/">
    adopet.top
  </a>
</div>

# About ADOPET

ADOPET was developed during the Software Engineering II course as part of an extension project, with the goal of positively impacting the community by applying the knowledge acquired at the university. </br>

After analyzing other pet adoption platforms, we were motivated to create a website that helps adopters find their ideal companions while addressing common adoption challenges, such as pet abandonment. To tackle this, we implemented a recommendation system that suggests pets based on user preferences. </br>
 
These preferences are collected through a questionnaire completed during the user registration process and are used to customize the recommendations using machine learning techniques.

# Details

Behind the scenes, ADOPET Rs uses a **Feature-Weighted Content-Based Filtering** approach. Similarities between users and pets are calculated using the cosine similarity function, which measures how closely the user's profile aligns with each pet's characteristics. </br>

The user's profile vector is multiplied by a vector of feature weights to emphasize the importance of certain attributes (such as expense range, animal preference, and housing size). Additionally, features with lower-bound limits undergo a value adjustment process. This allows the system to, for example, recommend a more affordable pet even if the user initially selected a higher expense range, thus increasing the chances of successful adoption.

The recommender system was tested using 50 synthetic user profiles and 200 synthetic animal profiles. We obtained a mean precision of `0.615` and a mean recall of `0.613`, indicating that the number of false positives and false negatives is nearly balanced.

Further evaluations and optimization techniques will be applied, with a particular focus on improving recall. The goal is to ensure that as many relevant recommendations as possible are presented to the user, even at the cost of including some less relevant ones.

# Built With
- [**NumPy**](https://numpy.org/doc/) - Fundamental package for numerical computing with support for arrays, matrices, and mathematical operations.
- [**Pandas**](https://pandas.pydata.org/docs/) - Powerful data structures for data manipulation and analysis, especially for tabular data (DataFrames).
- [**Scikit-Learn**](https://scikit-learn.org/) - Widely used library for machine learning: classification, regression, clustering, dimensionality reduction, and more.
- [**FastAPI**](https://fastapi.tiangolo.com/) - High-performance web framework for building APIs with automatic OpenAPI docs. Based on Pydantic & Starlette.
- [**SQLAlchemy**](https://docs.sqlalchemy.org/) - Powerful SQL toolkit and Object-Relational Mapping (ORM) library for Python.

# Contributions

This project and the other ones that are part of the ADOPET enviroment, were built alongside with my university collegues: [**Lucas Losekann**](https://github.com/lucaslosekann), [**João Fraga**](https://github.com/joaopedrofraga), [**Maria Martim**](https://github.com/MariaMartim).

If you encounter a bug or wish to suggest an improvement, please create a pull request with a descriptive title and a detailed description. Clearly document what was fixed or improved, and include any relevant comments to ensure proper project documentation.

# Other
- [**ADOPET Back-end**](https://github.com/lucaslosekann/adopet-backend)
- [**ADOPET Front-end**](https://github.com/lucaslosekann/adopet-frontend)


