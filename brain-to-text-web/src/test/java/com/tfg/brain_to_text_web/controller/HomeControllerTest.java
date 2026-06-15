package com.tfg.brain_to_text_web.controller;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.tfg.brain_to_text_web.model.Prediction;
import com.tfg.brain_to_text_web.model.Trial;
import com.tfg.brain_to_text_web.service.TrialService;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.mock.web.MockHttpSession;
import org.springframework.ui.ConcurrentModel;

@ExtendWith(MockitoExtension.class)
class HomeControllerTest {

    @Mock
    private TrialService trialService;

    @InjectMocks
    private HomeController homeController;

    @Test
    void indexReturnsHomeTemplate() {
        assertEquals("index", homeController.index());
    }

    @Test
    void trialsAddsPaginationAndFiltersToModel() {
        Trial trial = new Trial();
        Page<Trial> page = new PageImpl<>(List.of(trial));
        when(trialService.findTrialsPage("val", 2, 8)).thenReturn(page);
        when(trialService.findSplits()).thenReturn(List.of("test", "val"));
        ConcurrentModel model = new ConcurrentModel();

        String view = homeController.trials("val", 2, model);

        assertEquals("trials", view);
        assertEquals(List.of(trial), model.getAttribute("trials"));
        assertEquals(page, model.getAttribute("trialPage"));
        assertEquals(List.of("test", "val"), model.getAttribute("splits"));
        assertEquals("val", model.getAttribute("selectedSplit"));
    }

    @Test
    void detailUsesBaselineByDefaultWhenNoModelIsStoredInSession() {
        Trial trial = new Trial();
        Prediction baseline = new Prediction();
        baseline.setModelName("baseline");
        Prediction proposed = new Prediction();
        proposed.setModelName("proposed");
        trial.setPredictions(List.of(proposed, baseline));
        when(trialService.findTrialWithPredictions(12L)).thenReturn(trial);
        ConcurrentModel model = new ConcurrentModel();

        String view = homeController.detail(12L, new MockHttpSession(), model);

        assertEquals("trial-detail", view);
        assertEquals(trial, model.getAttribute("trial"));
        assertEquals("baseline", model.getAttribute("selectedModel"));
        assertEquals(baseline, model.getAttribute("prediction"));
    }

    @Test
    void detailUsesSelectedModelFromSession() {
        Trial trial = new Trial();
        Prediction proposed = new Prediction();
        proposed.setModelName("proposed");
        trial.setPredictions(List.of(proposed));
        when(trialService.findTrialWithPredictions(12L)).thenReturn(trial);
        MockHttpSession session = new MockHttpSession();
        session.setAttribute("selectedModel:12", "proposed");
        ConcurrentModel model = new ConcurrentModel();

        String view = homeController.detail(12L, session, model);

        assertEquals("trial-detail", view);
        assertEquals("proposed", model.getAttribute("selectedModel"));
        assertEquals(proposed, model.getAttribute("prediction"));
    }

    @Test
    void detailSetsNullPredictionWhenSelectedModelDoesNotExist() {
        Trial trial = new Trial();
        trial.setPredictions(List.of());
        when(trialService.findTrialWithPredictions(12L)).thenReturn(trial);
        ConcurrentModel model = new ConcurrentModel();

        String view = homeController.detail(12L, new MockHttpSession(), model);

        assertEquals("trial-detail", view);
        assertNull(model.getAttribute("prediction"));
    }

    @Test
    void selectModelStoresValidModelInSession() {
        MockHttpSession session = new MockHttpSession();

        String redirect = homeController.selectModel(12L, "proposed", session);

        assertEquals("redirect:/trials/12", redirect);
        assertEquals("proposed", session.getAttribute("selectedModel:12"));
    }

    @Test
    void selectModelFallsBackToBaselineWhenModelIsInvalid() {
        MockHttpSession session = new MockHttpSession();

        String redirect = homeController.selectModel(12L, "unknown", session);

        assertEquals("redirect:/trials/12", redirect);
        assertEquals("baseline", session.getAttribute("selectedModel:12"));
    }
}
